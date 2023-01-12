import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from transformers import BertPreTrainedModel, BertModel
from sklearn.covariance import EmpiricalCovariance, LedoitWolf


def _reg_loss(x1,x2, args,margin=0.5):
    
    x1_norm = F.normalize(x1,dim = 0) # batch-wise normalization
    x2_norm = F.normalize(x2,dim = 0)
    
    cor_mat = (x1_norm.t() @ x2_norm).clamp(min=1e-7)


    diag = torch.diagonal(cor_mat)
    margin_mask = (diag>margin).type(torch.uint8)

    loss = (diag*margin_mask).sum()/args.train.batch_size
    
    return loss



def reg_loss(tok_embeddings, args):

    total_reg_loss = 0
    
    raw_tok_embeddings, aug_tok_embeddings =  tok_embeddings
    
    for i in range(len(raw_tok_embeddings)-1):
        raw_hid = raw_tok_embeddings[i]
        raw_hid_next = raw_tok_embeddings[i+1]

        aug_hid = aug_tok_embeddings[i]
        aug_hid_next = aug_tok_embeddings[i+1]
        
        
        total_reg_loss += (_reg_loss(raw_hid,raw_hid_next, args) + _reg_loss(aug_hid,aug_hid_next, args)) / 2
        
    return total_reg_loss


def get_sim_mat(x):
    x = F.normalize(x, dim=1)
    return (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores

def Supervised_NT_xent(sim_matrix, labels, temperature=0.2, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
    labels = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)

    loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return loss



def mean_pooling(output, attention_mask):
    # output : B X seq_len X D
    # attention_mask : B x seq_len
    input_mask_expanded = attention_mask[:,0:].unsqueeze(-1).expand(output.size()).float()
    return torch.sum(output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LACL(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.lacl_config = args
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        # --------------------------------------------------------------- #
        self.num_layers = 12
        self.prj_dim = int (args.projection_dim/len(self.lacl_config.gp_layers))
        
        if self.lacl_config.gp_pooling == 'concat':
            self.global_projector = nn.Sequential(nn.Linear(768,args.encoder_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Linear(args.encoder_dim, self.prj_dim))
        elif self.lacl_config.gp_pooling == 'mean_pool':
            self.global_projector = nn.Sequential(nn.Linear(768,args.encoder_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Linear(args.encoder_dim, 768))
        
        # self.cls_projector = nn.Sequential(nn.Linear(768,args.encoder_dim),
        #                                nn.ReLU(),
        #                                nn.Dropout(0.1),
        #                                nn.Linear(args.encoder_dim, 128))
        # --------------------------------------------------------------- #
        self.init_weights()
        self.model_name = 'LACL_fixed'
        

    def forward(self, input_ids=None, attention_mask=None, index=None,labels=None,token_type_ids=None):
        
        return_dict = {}        
        # feed input
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # out_all_hidden = outputs[2]
        out_last_hidden = outputs.last_hidden_state
        # return_dict['all_hidden'] = out_all_hidden
        out_last_cls = out_last_hidden[:, 0, :]
        
        
        # last_cls
        return_dict['last_cls'] = out_last_cls

        # Global_projection & Reg_loss input
        lw_gp = []
        lw_mean_tok_embedding= []
        for i in range(1,1 + self.num_layers):    

            if i in self.lacl_config.gp_layers:

                if self.lacl_config.gp_location =='cls':
                    all_tokens = out_last_hidden[:,0,:]
                    lw_mean_tok_embedding.append(all_tokens)
                    lw_gp.append(self.global_projector(out_last_hidden[:,0,:])) 

                elif self.lacl_config.gp_location =='token':
                    pooled = mean_pooling(out_last_hidden[:,0:,:], attention_mask)
                    lw_mean_tok_embedding.append(pooled)
                    lw_gp.append(self.global_projector(pooled)) 

        
        if self.lacl_config.gp_pooling == 'concat':
            global_projection = torch.cat(lw_gp, dim=1)
        elif self.lacl_config.gp_pooling == 'mean_pool':
            global_projection = sum(lw_gp)/len(lw_gp)

        return_dict['lw_mean_embedding'] = lw_gp
        return_dict['global_projection'] = global_projection
        # breakpoint()

        return return_dict

    def compute_ood(self,input_ids=None,attention_mask=None,labels=None,indices=None,ind=False):
        
        b_dict = self(input_ids, attention_mask=attention_mask)
        pooled = b_dict['global_projection']

        maha_score = []
        for c in self.all_classes:
            centered_pooled = pooled - self.class_mean[c].unsqueeze(0)
            ms = torch.diag(centered_pooled @ self.class_var @ centered_pooled.t())
            maha_score.append(ms)
        maha_score = torch.stack(maha_score, dim=-1)
        maha_score, pred = maha_score.min(-1)
        maha_score = -maha_score
        
        if ind == True:
            correct = (labels == pred).float().sum()
        else:
            correct = 0

        norm_pooled = F.normalize(pooled, dim=-1)

        #! HS
        cosine_score = norm_pooled @ self.norm_bank.t()
        
        cosine_score, cosine_idx = cosine_score.topk(k=self.lacl_config.cosine_top_k,dim=-1)
        harmonic_weight = np.reciprocal([float(i) for i in range(1,1+self.lacl_config.cosine_top_k)])
        cosine_score = (cosine_score * torch.from_numpy(harmonic_weight).cuda()).sum(dim=-1)
        

        cosine_correct = sum(self.label_bank[[cosine_idx.squeeze()]] ==labels)

        ood_keys = {
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'maha_acc': correct,
            'cosine_correct': cosine_correct
             
        }
        return ood_keys

    def prepare_ood(self, dataloader=None):
        self.bank = None
        self.label_bank = None
        for batch in dataloader:
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            labels = batch['labels']
            
            b_dict = self(**batch)
            pooled = b_dict['global_projection']

            if self.bank is None:
                self.bank = pooled.clone().detach()
                self.label_bank = labels.clone().detach()
            else:
                bank = pooled.clone().detach()
                label_bank = labels.clone().detach()
                self.bank = torch.cat([bank, self.bank], dim=0)
                self.label_bank = torch.cat([label_bank, self.label_bank], dim=0)


        self.norm_bank = F.normalize(self.bank, dim=-1)
        N, d = self.bank.size()
        self.all_classes = list(set(self.label_bank.tolist()))
        self.class_mean = torch.zeros(max(self.all_classes) + 1, d).cuda()

        for c in self.all_classes:
            self.class_mean[c] = (self.bank[self.label_bank == c].mean(0))
        centered_bank = (self.bank - self.class_mean[self.label_bank]).detach().cpu().numpy()
        
        precision = LedoitWolf().fit(centered_bank).precision_.astype(np.float32)
        self.class_var = torch.from_numpy(precision).float().cuda()


from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        if key =='maha_acc':
            try:
                new_dict[key] = 0
                for i in l:
                    new_dict[key] += i[key]
            except:
                pass
        elif key =='cosine_correct':
            pass
        else:
            new_dict[key] = []
            for i in l:
                new_dict[key] += i[key]
    return new_dict


def evaluate_ood(args, model, features, ood, tag):

    keys = ['maha', 'cosine','maha_acc']
    dataloader = features
    # dataloader = DataLoader(features, batch_size=args.train_batch)
    in_scores = []
    
    cosine_correct, total_len = 0, 0
    for batch in dataloader:
        model.eval()
        batch = {key: value.cuda() for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch, ind = True)
            in_scores.append(ood_keys)
        cosine_correct += ood_keys['cosine_correct']
        total_len+=len(batch['labels'])

    cosine_ind_acc = float(cosine_correct/total_len)
        
    in_scores = merge_keys(in_scores, keys)
    
    dataloader = ood
    # dataloader = DataLoader(ood, batch_size=args.train_batch)
    out_scores = []
    out_labels_origin = []
    for batch in dataloader:
        model.eval()
        batch = {key: value.cuda() for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch)
            out_scores.append(ood_keys)
            out_labels_origin.extend(batch['labels'].tolist())
    out_scores = merge_keys(out_scores, keys)
    
    outputs = {}

    for key in keys:
        if key == 'maha_acc':
            outputs[tag+"_"+key] = float(in_scores[key] /len(features))
        else:
            ins = np.array(in_scores[key], dtype=np.float64)
            outs = np.array(out_scores[key], dtype=np.float64)
            inl = np.ones_like(ins).astype(np.int64)
            outl = np.zeros_like(outs).astype(np.int64)
            scores = np.concatenate([ins, outs], axis=0)
            labels = np.concatenate([inl, outl], axis=0)

            
            auroc = get_auroc(labels, scores)
            # fpr_95 = get_fpr_95(labels, scores, return_indices=False)
            
            fpr_95= fpr_at_95(ins,outs,inl,outl)
            
            
            outputs[tag + "_" + key + "_auroc"] = auroc
            outputs[tag + "_" + key + "_fpr95"] = fpr_95
        

    outputs['cosine_acc']=cosine_ind_acc
    return outputs

def fpr_at_95(ins,outs,inl,outl):
    # calculate the falsepositive error when tpr is 95%
    
    ins = sorted(ins)
    delta = ins[len(ins)//20]

    correct_index, wrong_index=[], []
    
    for idx, out in enumerate(outs):
        if out >= delta:
            wrong_index.append(idx)
        else:
            correct_index.append(idx)

    fpr = len(wrong_index) /len(outs)

    return fpr


def get_auroc(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    return roc_auc_score(new_key, prediction)
