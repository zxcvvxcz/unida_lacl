
import argparse
import time
import logging
import copy
import os

import easydict
import torch
import yaml
import numpy as np
import torch.backends.cudnn as cudnn
import pdb

from torch import (
    nn,
    optim
)
from tqdm import tqdm
from tensorboardX import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoConfig,
    get_scheduler,
)

from models.lacl import *
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything, parse_args
from utils.evaluation import HScore, Accuracy
from utils.data import get_dataloaders_for_oda_cl, ForeverDataIterator

cudnn.benchmark = True
cudnn.deterministic = True


logger = logging.getLogger(__name__)

def cheating_test(model, dataloader, unknown_class, is_cda=False, metric_name='total_accuracy', start=0.0, end=1.0, step=0.005):
    thresholds = list(np.arange(start, end, step))
    num_thresholds = len(thresholds)

    metric = HScore(unknown_class)

    print(f'Number of thresholds : {num_thresholds}')

    metrics = [copy.deepcopy(metric) for _ in range(num_thresholds)]

    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(dataloader, desc='Testing')):

            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            
            b_dict = model(**test_batch)
            pooled = b_dict['global_projection']
            
            norm_pooled = F.normalize(pooled, dim=-1)

            cosine_score = norm_pooled @ model.norm_bank.t()
            cosine_score, cosine_idx = cosine_score.topk(k=model.lacl_config.cosine_top_k,dim=-1)
            harmonic_weight = np.reciprocal([float(i) for i in range(1, 1+model.lacl_config.cosine_top_k)])
            cosine_score = (cosine_score * torch.from_numpy(harmonic_weight).cuda()).sum(dim=-1)
            
            weight = cosine_score
            predictions = model.label_bank[[cosine_idx.squeeze()]]

            # check for best threshold
            for index in range(num_thresholds):
                tmp_predictions = predictions.clone().detach()
                threshold = thresholds[index]

                # predict "unknown" for opda setting
                if not is_cda:
                    unknown = (weight < threshold).squeeze()
                    tmp_predictions[unknown] = unknown_class

                metrics[index].add_batch(
                    predictions=tmp_predictions,
                    references=labels
                )

    best_threshold = 0
    best_metric = 0
    best_results = None

    for index in range(num_thresholds):
        threshold = thresholds[index]

        results = metrics[index].compute()
        current_metric = results[metric_name] * 100

        if current_metric >= best_metric:
            best_metric = current_metric
            best_threshold = threshold
            best_results = results

    best_results['threshold'] = best_threshold

    return best_results


def test_for_cda(model, dataloader):
    metric = Accuracy()

    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(dataloader, desc='Testing CDA'):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']


            # predictions : (batch, )
            b_dict = model(**test_batch)
            pooled = b_dict['global_projection']
            
            norm_pooled = F.normalize(pooled, dim=-1)

            cosine_score = norm_pooled @ model.norm_bank.t()
            cosine_score, cosine_idx = cosine_score.topk(k=model.lacl_config.cosine_top_k,dim=-1)
            harmonic_weight = np.reciprocal([float(i) for i in range(1, 1+model.lacl_config.cosine_top_k)])
            cosine_score = (cosine_score * torch.from_numpy(harmonic_weight).cuda()).sum(dim=-1)
            
            predictions = model.label_bank[[cosine_idx.squeeze()]]

            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()

    return results

def test_with_threshold(model, dataloader, unknown_class, threshold):
    logger.info(f'Test with threshold {threshold}')
    metric = HScore(unknown_class)

    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(dataloader, desc='testing '):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            b_dict = model(**test_batch)
            pooled = b_dict['global_projection']
            
            norm_pooled = F.normalize(pooled, dim=-1)

            #! HS
            cosine_score = norm_pooled @ model.norm_bank.t()
            cosine_score, cosine_idx = cosine_score.topk(k=model.lacl_config.cosine_top_k,dim=-1)
            harmonic_weight = np.reciprocal([float(i) for i in range(1, 1+model.lacl_config.cosine_top_k)])
            cosine_score = (cosine_score * torch.from_numpy(harmonic_weight).cuda()).sum(dim=-1)
            
            weight = cosine_score
            predictions = model.label_bank[[cosine_idx.squeeze()]] 

            predictions[weight <= threshold] = unknown_class
            
            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()
    results['threshold'] = threshold

    return results

def eval_with_threshold(model, dataloader, is_cda, unknown_class, threshold):
    logger.info(f'Test with threshold {threshold}')
    # cosine_correct, total_len = 0, 0
    metric = Accuracy()
    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(dataloader, desc='testing '):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']
            
            b_dict = model(**test_batch)
            pooled = b_dict['global_projection']
            
            norm_pooled = F.normalize(pooled, dim=-1)

            #! HS
            cosine_score = norm_pooled @ model.norm_bank.t()
            cosine_score, cosine_idx = cosine_score.topk(k=model.lacl_config.cosine_top_k,dim=-1)
            harmonic_weight = np.reciprocal([float(i) for i in range(1, 1+model.lacl_config.cosine_top_k)])
            cosine_score = (cosine_score * torch.from_numpy(harmonic_weight).cuda()).sum(dim=-1)
            
            # if is_cda:
            #     predictions[weight <= threshold] = unknown_class
            
            weight = cosine_score
            predictions = model.label_bank[[cosine_idx.squeeze()]]
            predictions[weight <= threshold] = unknown_class
            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()
    results['threshold'] = threshold
    return results

def main(args, save_config):
    seed_everything(args.train.seed)

    source_domain = None
    target_domain = None
    coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'
    ## LOGGINGS ##
    log_dir = f'{args.log.output_dir}/{args.dataset.name}/lacl/oda/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    print(f'{log_dir=}')
    # init logger
    logger_init(logger, log_dir)
    # init tensorboard summarywriter
    writer = SummaryWriter(log_dir)
    # dump configs
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        f.write(yaml.dump(save_config))
    ## LOGGINGS ##


    # count known class / unknown class
    num_source_labels = args.dataset.num_source_class
    num_class = num_source_labels
    unknown_label = num_source_labels
    logger.info(f'Classify {num_source_labels} + 1 = {num_class+1} classes.\n\n')


    
    ## INIT TOKENIZER ##
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)

    ## GET DATALOADER ##    
    train_dataloader, comb_train_dataloader, _, _, _ = get_dataloaders_for_oda_cl(tokenizer=tokenizer, root_path=args.dataset.data_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, batch_size=args.train.batch_size, max_length=args.train.max_length, source=source_domain, target=target_domain)
    # _, train_unlabeled_dataloader, _, _, _ = get_dataloaders_cl(tokenizer=tokenizer, root_path=args.dataset.data_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, batch_size=args.train.unlabeled_batch_size, max_length=args.train.max_length, source=source_domain, target=target_domain)
    _, _, eval_dataloader, test_dataloader, source_test_dataloader = get_dataloaders_for_oda_cl(tokenizer=tokenizer, root_path=args.dataset.data_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, batch_size=args.test.batch_size, max_length=args.train.max_length, source=source_domain, target=target_domain)

    num_step_per_epoch = len(comb_train_dataloader)
    total_step = args.train.num_train_epochs * num_step_per_epoch
    logger.info(f'Total epoch {args.train.num_train_epochs}, steps per epoch {num_step_per_epoch}, total step {total_step}')


    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    config = AutoConfig.from_pretrained(args.model.model_name_or_path, cache_dir=args.dataset.model_path, num_labels=num_class)
    # config.output_attentions = True
    # config.output_hidden_states = True
    model = LACL.from_pretrained(
        args.model.model_name_or_path,
        config=config,
        args=args,
    )
    model.cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total number of trained parameters : {num_total_params}')
    ## INIT MODEL ##

    ## OPTIMIZER & SCHEDULER ##  
    
    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.train.lr, eps=args.adam_epsilon)

    num_warmup_steps = int(total_step * 0.3)

    lr_scheduler = get_scheduler(
        name=args.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_step
    )
    ## OPTIMIZER & SCHEDULER ##  
    

    # data iter
    source_iter = ForeverDataIterator(comb_train_dataloader)
    # target_iter = ForeverDataIterator(train_unlabeled_dataloader)

    '''
    Always output logits. Therefore,
    BCE: BCEWithLogitsLoss
    CE: CrossEntropyLoss
    '''
    # CE-loss for classification
    # ce = nn.CrossEntropyLoss().cuda()
    # BCE-loss for domain classification
    # bce = nn.BCEWithLogitsLoss().cuda() 

    logger.info('Start main training...')

    global_step = 0
    current_epoch = 0
    best_acc = 0
    best_results = None
    early_stop_count = 0

    ## START TRAINING ##
    if args.train.train:
        logger.info(f'Start Training....')
        start_time = time.time()
        for current_epoch in range(1, args.train.num_train_epochs+1):
            model.train()
            total_cl_loss = total_reg_loss = 0

            # check early stop.
            if early_stop_count == args.train.early_stop:
                logger.info('Early stop. End.')
                break
            
            with tqdm(range(num_step_per_epoch)) as pbar:
                for current_step in pbar:

                    global_step += 1
                    # optimizer zero-grad
                    optimizer.zero_grad()

                    ####################
                    #                  #
                    #     Load Data    #
                    #                  #
                    ####################

                    ## source to cuda
                    source_batch1, source_batch2 = next(source_iter)
                    source_batch1 = {k: v.cuda() for k, v in source_batch1.items()}
                    source_batch2 = {k: v.cuda() for k, v in source_batch2.items()}
                    source_label = source_batch1['labels']
                    if global_step == 1:
                        print('Sanity check for data pair...')
                        print(f'Batch1: {tokenizer.batch_decode(source_batch1["input_ids"], skip_special_tokens=True)}')
                        print(f'Batch2: {tokenizer.batch_decode(source_batch2["input_ids"], skip_special_tokens=True)}')
                    ## target to cuda
                    # target_batch = next(target_iter)
                    # target_batch = {k: v.cuda() for k, v in target_batch.items()}

                    ####################
                    #                  #
                    #   Forward Pass   #
                    #                  #
                    ####################

                    ## source 
                    source_output1 = model(**source_batch1)
                    source_output2 = model(**source_batch2)
                    ## target 
                    # _, target_disc_output = model(**target_batch)
                    
                    
                    ####################
                    #                  #
                    #   Compute Loss   #
                    #                  #
                    ####################
                  
                    
                    cont = torch.cat([source_output1['global_projection'], source_output2['global_projection']], dim=0)
                    sim_mat = get_sim_mat(cont)
                    loss_cl_gp = Supervised_NT_xent(sim_mat, labels=source_label, temperature=args.temperature)
                    total_cl_loss += loss_cl_gp.item()
                    
                    
                    tok_embeddings = [source_output1['lw_mean_embedding'], source_output2['lw_mean_embedding']]
                    loss_reg = reg_loss(tok_embeddings,args)
                    total_reg_loss += loss_reg.item()
                    
                    
                    loss = loss_cl_gp + loss_reg * args.reg_loss_weight
                    

                    # backward, optimization
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    
                    # write to tensorboard
                    writer.add_scalar('train/loss_cl_gp', loss_cl_gp, global_step)
                    writer.add_scalar('train/loss_reg', loss_reg, global_step)
                    pbar.set_description(f'TRAIN EPOCH {current_epoch}, CL:{total_cl_loss / (current_step+1):.2E}, REG:{total_reg_loss / (current_step+1):.2E}')
            ####################
            #                  #
            #     Evaluate     #
            #                  #
            ####################
            
            logger.info(f'Evaluate model at epoch {current_epoch} ...')

            # find optimal threshold from evaluation set (source domain) -> sub-optimal threshold
            model.prepare_ood(train_dataloader)
                
            results = eval_with_threshold(model, eval_dataloader, False, unknown_label, args.test.threshold)
            # write to tensorboard
            for k,v in results.items():
                writer.add_scalar(f'eval/{k}', v, global_step)
            
            
            if results['accuracy'] >= best_acc:
                # if results['accuracy'] > best_acc:
                early_stop_count = 0
                best_acc = results['accuracy']
                best_results = results

                print_dict(logger, string=f'\n* BEST TOTAL ACCURACY at epoch {current_epoch}', dict=results)
                logger.info('Saving best model...')
                torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))
                logger.info('Done saving...')

            else:
                print_dict(logger, string=f'\n* ACCURACY at epoch {current_epoch}', dict=results)
                logger.info('\nNot best. Pass.')

                early_stop_count += 1
                logger.info(f'Early stopping : {early_stop_count} / {args.train.early_stop}')
            # logger.info('TEST ON CDA SETTING.')
            # results = test_for_cda(model, test_dataloader)
            # for k,v in results.items():
            #     writer.add_scalar(f'test/{k}', v, 0)

            # print_dict(logger, string=f'\n\n** FINAL TARGET DOMAIN TEST RESULT', dict=results)
        
        end_time = time.time()
        logger.info(f'Done training full step. Total time : {end_time-start_time}')

        # skip evaluation with low accuracy
        if best_results['accuracy'] < 85:
            logger.info(f'Low total accuracy {best_results["accuracy"]}. Skip testing.')
            exit() 
        
            
    else:
        logger.info('Skip training... ')
    
    ####################
    #                  #
    #       Test       #
    #                  #
    ####################

    logger.info('Loading best model ...')
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best.pth')))
            
    logger.info('Test model...')
    logger.info('TEST WITH "UNKNOWN" CLASS.')
    best_threshold = best_results['threshold'] if best_results is not None else args.test.threshold
    results = test_with_threshold(model, test_dataloader, unknown_label, best_threshold)
    for k,v in results.items():
        writer.add_scalar(f'test/{k}', v, 0)

    print_dict(logger, string=f'\n\n** FINAL TARGET DOMAIN TEST RESULT', dict=results)

    # Find optimal threshold from test set (Cheating)
    # find model with the best h-score
    results = cheating_test(model, test_dataloader, unknown_label, metric_name='h_score', start=args.test.min_threshold, end=args.test.max_threshold, step=args.test.step)
    # write to tensorboard
    for k,v in results.items():
        writer.add_scalar(f'test/{k}', v, 1)
    print_dict(logger, string=f'\n\n** CHEATING TARGET DOMAIN TEST RESULT', dict=results)


    logger.info('Done.')
                



if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

