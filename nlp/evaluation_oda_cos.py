
import time
import logging
import copy
import os

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pdb

from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.covariance import LedoitWolf

from models import (
    bert,
    cmu, 
    dann, 
    ovanet, 
    uan,
    udalm
)
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything, parse_args
from utils.evaluation import HScore
from utils.data import get_dataloaders_for_oda

cudnn.benchmark = True
cudnn.deterministic = True

logger = logging.getLogger(__name__)

METHOD_TO_MODEL = {
    'fine_tuning' : bert.BERT,
    'dann' : dann.DANN,
    'uan' : uan.UAN,
    'cmu' : cmu.CMU,
    'ovanet' : ovanet.OVANET,
    'udalm' : udalm.UDALM,
}


def cheating_test(model, dataloader, output_dict, unknown_class, metric_name='total_accuracy', start=0.0, end=1.0, step=0.005):
    thresholds = list(np.arange(start, end, step))
    num_thresholds = len(thresholds)

    metric = HScore(unknown_class)

    print(f'Number of thresholds : {num_thresholds}')

    cosine_metrics = [copy.deepcopy(metric) for _ in range(num_thresholds)]

    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(dataloader, desc='Testing')):

            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            # shape : (batch, hidden_dim)
            embeddings = model(**test_batch, embeddings_only=True)

            norm_embeddings = F.normalize(embeddings, dim=-1)

            ## COSINE ##
            # shape : (batch, hidden_dim) @ (hidden_dim, num_samples) -> (batch, num_samples)
            cosine_scores = norm_embeddings @ output_dict['norm_bank'].t()

            # shape : (batch, )
            cosine_score, max_indices = cosine_scores.max(-1)
            cosine_pred = output_dict['label_bank'][max_indices]


            # check for best threshold
            for index in range(num_thresholds):
                tmp_cosine_predictions = cosine_pred.clone().detach()

                threshold = thresholds[index]

                unknown = (cosine_score < threshold).squeeze()
                tmp_cosine_predictions[unknown] = unknown_class

                cosine_metrics[index].add_batch(
                    predictions=tmp_cosine_predictions,
                    references=labels
                )

    best_threshold = 0
    best_metric = 0
    best_results = None

    for index in range(num_thresholds):
        threshold = thresholds[index]

        results = cosine_metrics[index].compute()
        current_metric = results[metric_name] * 100

        if current_metric >= best_metric:
            best_metric = current_metric
            best_threshold = threshold
            best_results = results

    best_results['threshold'] = best_threshold
    
    return best_results

def prepare_ood(model, dataloader=None):
    bank = None
    label_bank = None
    for batch in tqdm(dataloader, desc='Generating training distribution'):
        model.eval()
        batch = {key: value.cuda() for key, value in batch.items()}
        labels = batch['labels']
        
        # shape : (batch, hidden_dim)
        pooled = model.forward(**batch, embeddings_only=True)

        if bank is None:
            bank = pooled.clone().detach()
            label_bank = labels.clone().detach()
        else:
            new_bank = pooled.clone().detach()
            new_label_bank = labels.clone().detach()
            bank = torch.cat([new_bank, bank], dim=0)
            label_bank = torch.cat([new_label_bank, label_bank], dim=0)


    # shape : (num_sample, hidden_dim)
    norm_bank = F.normalize(bank, dim=-1)
    # shape : (num_sample, hidden_dim)
    N, d = bank.size()
    # shape : (num_class, )
    all_classes = list(set(label_bank.tolist()))
    # shape : (num_class, hidden_dim)
    class_mean = torch.zeros(max(all_classes) + 1, d).cuda()

    for c in all_classes:
        class_mean[c] = (bank[label_bank == c].mean(0))
    # shape : (num_class, hidden_dim)
    centered_bank = (bank - class_mean[label_bank]).detach().cpu().numpy()
    
    # precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
    precision = LedoitWolf().fit(centered_bank).precision_.astype(np.float32)
    class_var = torch.from_numpy(precision).float().cuda()

    return {
        # shape : (hidden_dim, hidden_dim)
        'class_var' : class_var,
        # shape : (num_class, hidden_dim)
        'class_mean' : class_mean,
        # shape : (num_samples, hidden_dim)
        'norm_bank' : norm_bank,
        # list of range(0, num_class)
        'all_classes' : all_classes,
        # shape : (num_class, )
        'label_bank' : label_bank,
    }


def test_with_threshold(model, dataloader, output_dict, unknown_class, threshold):
    logger.info(f'Test with threshold {threshold}')
  
    metric = HScore(unknown_class)

    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(dataloader, desc='Testing')):

            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            # shape : (batch, hidden_dim)
            embeddings = model(**test_batch, embeddings_only=True)

            norm_embeddings = F.normalize(embeddings, dim=-1)

            ## COSINE ##
            # shape : (batch, hidden_dim) @ (hidden_dim, num_samples) -> (batch, num_samples)
            cosine_scores = norm_embeddings @ output_dict['norm_bank'].t()

            # shape : (batch, )
            cosine_score, max_indices = cosine_scores.max(-1)
            cosine_pred = output_dict['label_bank'][max_indices]

            cosine_pred[cosine_score < threshold] = unknown_class

            metric.add_batch(predictions=cosine_pred, references=labels)

    results = metric.compute()
    results['threshold'] = threshold

    return results


def get_cosine_scores(model, dataloader, output_dict):
    max_cosine_list = []

    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(dataloader, desc='Testing')):

            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            # shape : (batch, hidden_dim)
            embeddings = model(**test_batch, embeddings_only=True)

            norm_embeddings = F.normalize(embeddings, dim=-1)

            ## COSINE ##
            # shape : (batch, hidden_dim) @ (hidden_dim, num_samples) -> (batch, num_samples)
            cosine_scores = norm_embeddings @ output_dict['norm_bank'].t()

            # shape : (batch, )
            cosine_score, _ = cosine_scores.max(-1)

            max_cosine_list.append(cosine_score)
        
    max_cosine_list = torch.concat(max_cosine_list)

    return max_cosine_list

def main(args, save_config):
    seed_everything(args.train.seed)

    assert args.method_name in METHOD_TO_MODEL.keys()
    
    ## LOGGINGS ##
    log_dir = f'{args.log.output_dir}/{args.dataset.name}/{args.method_name}/oda/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    
    # init logger
    logger_init(logger, log_dir)
    ## LOGGINGS ##


    # count known class / unknown class
    num_source_labels = args.dataset.num_source_class
    num_class = num_source_labels
    unknown_label = num_source_labels
    logger.info(f'Classify {num_source_labels} + 1 = {num_class+1} classes.\n\n')

    
    ## INIT TOKENIZER ##
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)

    ## GET DATALOADER ##
    train_dataloader, _, eval_dataloader, test_dataloader, source_test_dataloader = get_dataloaders_for_oda(tokenizer=tokenizer, root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, batch_size=args.test.batch_size, max_length=args.train.max_length)

    ## INIT MODEL ##
    logger.info(f'Init model {args.method_name} ...')
    METHOD = METHOD_TO_MODEL.get(args.method_name)
    start_time = time.time()
    model = METHOD(
        model_name=args.model.model_name_or_path,
        num_class=num_class,
        max_train_step=100,     # dummy value
    ).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total number of trained parameters : {num_total_params}')
    ## INIT MODEL ##

     
    model_dir = os.path.join(log_dir, 'best.pth')
    logger.info(f'Loading best model from : {model_dir}')
    model.load_state_dict(torch.load(model_dir))

    
    ####################
    #                  #
    #       Test       #
    #                  #
    ####################
    
    # get hidden representations
    output_dict = prepare_ood(model, dataloader=train_dataloader)

    # test with cosine similarity
    best_results = cheating_test(model, test_dataloader, output_dict, unknown_label, metric_name='h_score', start=0.0, end=1.0, step=0.005)

    print_dict(logger, string=f'\n\n** Cheating target domain test result using COSINE SIMILARITY', dict=best_results)

    logger.info('Get Cosine Similarity Scores from train set')
    max_cosine_list = get_cosine_scores(model, train_dataloader, output_dict)

    score_count = len(max_cosine_list)
    logger.info(f'Total count : {score_count}')

    sorted_cosine_scores, _ = torch.sort(max_cosine_list, descending=True)
    logger.info(f'Get H-score @ {args.test.fpr_rate}')
    threshold_index = round(score_count * args.test.fpr_rate)
    threshold = sorted_cosine_scores[threshold_index]

    logger.info(f'* H-score @ {args.test.fpr_rate} with threshold {threshold}...')
    results = test_with_threshold(model, test_dataloader, output_dict, unknown_label, threshold)

    print_dict(logger, string=f'\n\n** COSINE SIMILARITY @ {args.test.fpr_rate}', dict=results)

    
    logger.info('Done.')

if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

