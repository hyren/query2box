#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import Query2box
from dataloader import *
from tensorboardX import SummaryWriter
import time
import pickle
import collections

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=50000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    parser.add_argument('--geo', default='vec', type=str, help='vec or box')
    parser.add_argument('--print_on_screen', action='store_true')
    
    parser.add_argument('--task', default='1c.2c.3c.2i.3i', type=str)
    parser.add_argument('--stepsforpath', type=int, default=0)

    parser.add_argument('--offset_deepsets', default='vanilla', type=str, help='inductive or vanilla or min')
    parser.add_argument('--offset_use_center', action='store_true')
    parser.add_argument('--center_deepsets', default='vanilla', type=str, help='vanilla or attention or mean')
    parser.add_argument('--center_use_offset', action='store_true')
    parser.add_argument('--entity_use_offset', action='store_true')
    parser.add_argument('--att_reg', default=0.0, type=float)
    parser.add_argument('--off_reg', default=0.0, type=float)
    parser.add_argument('--att_tem', default=1.0, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gamma2', default=0, type=float)
    parser.add_argument('--train_onehop_only', action='store_true')
    parser.add_argument('--center_reg', default=0.0, type=float, help='alpha in the paper')
    parser.add_argument('--bn', default='no', type=str, help='no or before or after')
    parser.add_argument('--n_att', type=int, default=1)
    parser.add_argument('--activation', default='relu', type=str, help='relu or none or softplus')

    return parser.parse_args(args)

def override_config(args): #! may update here
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args, before_finetune=False):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json' if not before_finetune else 'config_before.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint' if not before_finetune else 'checkpoint_before')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding' if not before_finetune else 'entity_embedding_before'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding' if not before_finetune else 'relation_embedding_before'), 
        relation_embedding
    )

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
def main(args):
    set_global_seed(args.seed)
    args.test_batch_size = 1
    assert args.bn in ['no', 'before', 'after']
    assert args.n_att >= 1 and args.n_att <= 3
    assert args.max_steps == args.stepsforpath
    if args.geo == 'box':
        assert 'Box' in args.model
    elif args.geo == 'vec':
        assert 'Box' not in args.model
        
    if args.train_onehop_only:
        assert '1c' in args.task
        args.center_deepsets = 'mean'
        if args.geo == 'box':
            args.offset_deepsets = 'min'

    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.evaluate_train):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    # if args.do_train and args.save_path is None:
    #     raise ValueError('Where do you want to save your trained model?')

    cur_time = parse_time()
    print ("overide save string.")
    if args.task == '1c':
        args.stepsforpath = 0
    else:
        assert args.stepsforpath <= args.max_steps
    # logs_newfb237_inter
    
    args.save_path = 'logs/%s/%s/'%(args.data_path.split('/')[-1], args.geo)
    writer = SummaryWriter(args.save_path)
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    set_logger(args)

    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Geo: %s' % args.geo)
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('#stepsforpath: %d' % args.stepsforpath)

    tasks = args.task.split('.')
    
    train_ans = dict()
    valid_ans = dict()
    valid_ans_hard = dict()
    test_ans = dict()
    test_ans_hard = dict()

    if '1c' in tasks:
        with open('%s/train_triples_1c.pkl'%args.data_path, 'rb') as handle:
            train_triples = pickle.load(handle)
        with open('%s/train_ans_1c.pkl'%args.data_path, 'rb') as handle:
            train_ans_1 = pickle.load(handle)
        with open('%s/valid_triples_1c.pkl'%args.data_path, 'rb') as handle:
            valid_triples = pickle.load(handle)
        with open('%s/valid_ans_1c.pkl'%args.data_path, 'rb') as handle:
            valid_ans_1 = pickle.load(handle)
        with open('%s/valid_ans_1c_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_1_hard = pickle.load(handle)
        with open('%s/test_triples_1c.pkl'%args.data_path, 'rb') as handle:
            test_triples = pickle.load(handle)
        with open('%s/test_ans_1c.pkl'%args.data_path, 'rb') as handle:
            test_ans_1 = pickle.load(handle)
        with open('%s/test_ans_1c_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_1_hard = pickle.load(handle)
        train_ans.update(train_ans_1)
        valid_ans.update(valid_ans_1)
        valid_ans_hard.update(valid_ans_1_hard)
        test_ans.update(test_ans_1)
        test_ans_hard.update(test_ans_1_hard)

    if '2c' in tasks:
        with open('%s/train_triples_2c.pkl'%args.data_path, 'rb') as handle:
            train_triples_2 = pickle.load(handle)
        with open('%s/train_ans_2c.pkl'%args.data_path, 'rb') as handle:
            train_ans_2 = pickle.load(handle)
        with open('%s/valid_triples_2c.pkl'%args.data_path, 'rb') as handle:
            valid_triples_2 = pickle.load(handle)
        with open('%s/valid_ans_2c.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2 = pickle.load(handle)
        with open('%s/valid_ans_2c_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2_hard = pickle.load(handle)
        with open('%s/test_triples_2c.pkl'%args.data_path, 'rb') as handle:
            test_triples_2 = pickle.load(handle)
        with open('%s/test_ans_2c.pkl'%args.data_path, 'rb') as handle:
            test_ans_2 = pickle.load(handle)
        with open('%s/test_ans_2c_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_2_hard = pickle.load(handle)
        train_ans.update(train_ans_2)
        valid_ans.update(valid_ans_2)
        valid_ans_hard.update(valid_ans_2_hard)
        test_ans.update(test_ans_2)
        test_ans_hard.update(test_ans_2_hard)

    if '3c' in tasks:
        with open('%s/train_triples_3c.pkl'%args.data_path, 'rb') as handle:
            train_triples_3 = pickle.load(handle)
        with open('%s/train_ans_3c.pkl'%args.data_path, 'rb') as handle:
            train_ans_3 = pickle.load(handle)
        with open('%s/valid_triples_3c.pkl'%args.data_path, 'rb') as handle:
            valid_triples_3 = pickle.load(handle)
        with open('%s/valid_ans_3c.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3 = pickle.load(handle)
        with open('%s/valid_ans_3c_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3_hard = pickle.load(handle)
        with open('%s/test_triples_3c.pkl'%args.data_path, 'rb') as handle:
            test_triples_3 = pickle.load(handle)
        with open('%s/test_ans_3c.pkl'%args.data_path, 'rb') as handle:
            test_ans_3 = pickle.load(handle)
        with open('%s/test_ans_3c_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_3_hard = pickle.load(handle)
        train_ans.update(train_ans_3)
        valid_ans.update(valid_ans_3)
        valid_ans_hard.update(valid_ans_3_hard)
        test_ans.update(test_ans_3)
        test_ans_hard.update(test_ans_3_hard)

    if '2i' in tasks:
        with open('%s/train_triples_2i.pkl'%args.data_path, 'rb') as handle:
            train_triples_2i = pickle.load(handle)
        with open('%s/train_ans_2i.pkl'%args.data_path, 'rb') as handle:
            train_ans_2i = pickle.load(handle)
        with open('%s/valid_triples_2i.pkl'%args.data_path, 'rb') as handle:
            valid_triples_2i = pickle.load(handle)
        with open('%s/valid_ans_2i.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2i = pickle.load(handle)
        with open('%s/valid_ans_2i_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2i_hard = pickle.load(handle)
        with open('%s/test_triples_2i.pkl'%args.data_path, 'rb') as handle:
            test_triples_2i = pickle.load(handle)
        with open('%s/test_ans_2i.pkl'%args.data_path, 'rb') as handle:
            test_ans_2i = pickle.load(handle)
        with open('%s/test_ans_2i_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_2i_hard = pickle.load(handle)
        train_ans.update(train_ans_2i)
        valid_ans.update(valid_ans_2i)
        valid_ans_hard.update(valid_ans_2i_hard)
        test_ans.update(test_ans_2i)
        test_ans_hard.update(test_ans_2i_hard)

    if '3i' in tasks:
        with open('%s/train_triples_3i.pkl'%args.data_path, 'rb') as handle:
            train_triples_3i = pickle.load(handle)
        with open('%s/train_ans_3i.pkl'%args.data_path, 'rb') as handle:
            train_ans_3i = pickle.load(handle)
        with open('%s/valid_triples_3i.pkl'%args.data_path, 'rb') as handle:
            valid_triples_3i = pickle.load(handle)
        with open('%s/valid_ans_3i.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3i = pickle.load(handle)
        with open('%s/valid_ans_3i_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_3i_hard = pickle.load(handle)
        with open('%s/test_triples_3i.pkl'%args.data_path, 'rb') as handle:
            test_triples_3i = pickle.load(handle)
        with open('%s/test_ans_3i.pkl'%args.data_path, 'rb') as handle:
            test_ans_3i = pickle.load(handle)
        with open('%s/test_ans_3i_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_3i_hard = pickle.load(handle)
        train_ans.update(train_ans_3i)
        valid_ans.update(valid_ans_3i)
        valid_ans_hard.update(valid_ans_3i_hard)
        test_ans.update(test_ans_3i)
        test_ans_hard.update(test_ans_3i_hard)

    if 'ci' in tasks:
        with open('%s/valid_triples_ci.pkl'%args.data_path, 'rb') as handle:
            valid_triples_ci = pickle.load(handle)
        with open('%s/valid_ans_ci.pkl'%args.data_path, 'rb') as handle:
            valid_ans_ci = pickle.load(handle)
        with open('%s/valid_ans_ci_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_ci_hard = pickle.load(handle)
        with open('%s/test_triples_ci.pkl'%args.data_path, 'rb') as handle:
            test_triples_ci = pickle.load(handle)
        with open('%s/test_ans_ci.pkl'%args.data_path, 'rb') as handle:
            test_ans_ci = pickle.load(handle)
        with open('%s/test_ans_ci_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_ci_hard = pickle.load(handle)
        valid_ans.update(valid_ans_ci)
        valid_ans_hard.update(valid_ans_ci_hard)
        test_ans.update(test_ans_ci)
        test_ans_hard.update(test_ans_ci_hard)

    if 'ic' in tasks:
        with open('%s/valid_triples_ic.pkl'%args.data_path, 'rb') as handle:
            valid_triples_ic = pickle.load(handle)
        with open('%s/valid_ans_ic.pkl'%args.data_path, 'rb') as handle:
            valid_ans_ic = pickle.load(handle)
        with open('%s/valid_ans_ic_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_ic_hard = pickle.load(handle)
        with open('%s/test_triples_ic.pkl'%args.data_path, 'rb') as handle:
            test_triples_ic = pickle.load(handle)
        with open('%s/test_ans_ic.pkl'%args.data_path, 'rb') as handle:
            test_ans_ic = pickle.load(handle)
        with open('%s/test_ans_ic_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_ic_hard = pickle.load(handle)
        valid_ans.update(valid_ans_ic)
        valid_ans_hard.update(valid_ans_ic_hard)
        test_ans.update(test_ans_ic)
        test_ans_hard.update(test_ans_ic_hard)

    if 'uc' in tasks:
        with open('%s/valid_triples_uc.pkl'%args.data_path, 'rb') as handle:
            valid_triples_uc = pickle.load(handle)
        with open('%s/valid_ans_uc.pkl'%args.data_path, 'rb') as handle:
            valid_ans_uc = pickle.load(handle)
        with open('%s/valid_ans_uc_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_uc_hard = pickle.load(handle)
        with open('%s/test_triples_uc.pkl'%args.data_path, 'rb') as handle:
            test_triples_uc = pickle.load(handle)
        with open('%s/test_ans_uc.pkl'%args.data_path, 'rb') as handle:
            test_ans_uc = pickle.load(handle)
        with open('%s/test_ans_uc_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_uc_hard = pickle.load(handle)
        valid_ans.update(valid_ans_uc)
        valid_ans_hard.update(valid_ans_uc_hard)
        test_ans.update(test_ans_uc)
        test_ans_hard.update(test_ans_uc_hard)

    if '2u' in tasks:
        with open('%s/valid_triples_2u.pkl'%args.data_path, 'rb') as handle:
            valid_triples_2u = pickle.load(handle)
        with open('%s/valid_ans_2u.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2u = pickle.load(handle)
        with open('%s/valid_ans_2u_hard.pkl'%args.data_path, 'rb') as handle:
            valid_ans_2u_hard = pickle.load(handle)
        with open('%s/test_triples_2u.pkl'%args.data_path, 'rb') as handle:
            test_triples_2u = pickle.load(handle)
        with open('%s/test_ans_2u.pkl'%args.data_path, 'rb') as handle:
            test_ans_2u = pickle.load(handle)
        with open('%s/test_ans_2u_hard.pkl'%args.data_path, 'rb') as handle:
            test_ans_2u_hard = pickle.load(handle)
        valid_ans.update(valid_ans_2u)
        valid_ans_hard.update(valid_ans_2u_hard)
        test_ans.update(test_ans_2u)
        test_ans_hard.update(test_ans_2u_hard)

    if '1c' in tasks:
        logging.info('#train: %d' % len(train_triples))
        logging.info('#valid: %d' % len(valid_triples))
        logging.info('#test: %d' % len(test_triples))
    
    if '2c' in tasks:
        logging.info('#train_2c: %d' % len(train_triples_2))
        logging.info('#valid_2c: %d' % len(valid_triples_2))
        logging.info('#test_2c: %d' % len(test_triples_2))
    
    if '3c' in tasks:
        logging.info('#train_3c: %d' % len(train_triples_3))
        logging.info('#valid_3c: %d' % len(valid_triples_3))
        logging.info('#test_3c: %d' % len(test_triples_3))
    
    if '2i' in tasks:
        logging.info('#train_2i: %d' % len(train_triples_2i))
        logging.info('#valid_2i: %d' % len(valid_triples_2i))
        logging.info('#test_2i: %d' % len(test_triples_2i))
    
    if '3i' in tasks:
        logging.info('#train_3i: %d' % len(train_triples_3i))
        logging.info('#valid_3i: %d' % len(valid_triples_3i))
        logging.info('#test_3i: %d' % len(test_triples_3i))
    
    if 'ci' in tasks:
        logging.info('#valid_ci: %d' % len(valid_triples_ci))
        logging.info('#test_ci: %d' % len(test_triples_ci))
    
    if 'ic' in tasks:
        logging.info('#valid_ic: %d' % len(valid_triples_ic))
        logging.info('#test_ic: %d' % len(test_triples_ic))

    if '2u' in tasks:
        logging.info('#valid_2u: %d' % len(valid_triples_2u))
        logging.info('#test_2u: %d' % len(test_triples_2u))

    if 'uc' in tasks:
        logging.info('#valid_uc: %d' % len(valid_triples_uc))
        logging.info('#test_uc: %d' % len(test_triples_uc))


    query2box = Query2box(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        writer=writer,
        geo=args.geo,
        cen=args.center_reg,
        offset_deepsets = args.offset_deepsets,
        center_deepsets = args.center_deepsets,
        offset_use_center = args.offset_use_center,
        center_use_offset = args.center_use_offset,
        att_reg = args.att_reg,
        off_reg = args.off_reg,
        att_tem = args.att_tem,
        euo = args.entity_use_offset,
        gamma2 = args.gamma2,
        bn = args.bn,
        nat = args.n_att,
        activation = args.activation
    )
    
    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in query2box.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        query2box = query2box.cuda()
    
    if args.do_train:
        # Set training dataloader iterator
        if '1c' in tasks:
            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainDataset.collate_fn
            )
            train_iterator = SingledirectionalOneShotIterator(train_dataloader_tail, train_triples[0][-1])

        if '2c' in tasks:
            train_dataloader_2_tail = DataLoader(
                TrainDataset(train_triples_2, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainDataset.collate_fn
            )
            train_iterator_2 = SingledirectionalOneShotIterator(train_dataloader_2_tail, train_triples_2[0][-1])

        if '3c' in tasks:
            train_dataloader_3_tail = DataLoader(
                TrainDataset(train_triples_3, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainDataset.collate_fn
            )
            train_iterator_3 = SingledirectionalOneShotIterator(train_dataloader_3_tail, train_triples_3[0][-1])

        if '2i' in tasks:
            train_dataloader_2i_tail = DataLoader(
                TrainInterDataset(train_triples_2i, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn
            )
            train_iterator_2i = SingledirectionalOneShotIterator(train_dataloader_2i_tail, train_triples_2i[0][-1])

        if '3i' in tasks:
            train_dataloader_3i_tail = DataLoader(
                TrainInterDataset(train_triples_3i, nentity, nrelation, args.negative_sample_size, train_ans, 'tail-batch'), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num),
                collate_fn=TrainInterDataset.collate_fn
            )
            train_iterator_3i = SingledirectionalOneShotIterator(train_dataloader_3i_tail, train_triples_3i[0][-1])
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, query2box.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        query2box.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step 

    logging.info('task = %s' % args.task)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    def evaluate_test():
        average_metrics = collections.defaultdict(list)
        average_c_metrics = collections.defaultdict(list)
        average_c2_metrics = collections.defaultdict(list)
        average_i_metrics = collections.defaultdict(list)
        average_ex_metrics = collections.defaultdict(list)
        average_u_metrics = collections.defaultdict(list)
        if '2i' in tasks:
            metrics = query2box.test_step(query2box, test_triples_2i, test_ans, test_ans_hard, args)
            log_metrics('Test 2i', step, metrics)
            for metric in metrics:
                writer.add_scalar('Test_2i_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_i_metrics[metric].append(metrics[metric])
        if '3i' in tasks:
            metrics = query2box.test_step(query2box, test_triples_3i, test_ans, test_ans_hard, args)
            log_metrics('Test 3i', step, metrics)
            for metric in metrics:
                writer.add_scalar('Test_3i_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_i_metrics[metric].append(metrics[metric])
        if '2c' in tasks:
            metrics = query2box.test_step(query2box, test_triples_2, test_ans, test_ans_hard, args)
            log_metrics('Test 2c', step, metrics)
            for metric in metrics:
                writer.add_scalar('Test_2c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])
                average_c2_metrics[metric].append(metrics[metric])
        if '3c' in tasks:
            metrics = query2box.test_step(query2box, test_triples_3, test_ans, test_ans_hard, args)
            log_metrics('Test 3c', step, metrics)
            for metric in metrics:
                writer.add_scalar('Test_3c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])
                average_c2_metrics[metric].append(metrics[metric])
        if '1c' in tasks:
            metrics = query2box.test_step(query2box, test_triples, test_ans, test_ans_hard, args)
            log_metrics('Test 1c', step, metrics)
            for metric in metrics:
                writer.add_scalar('Test_1c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])
        if 'ci' in tasks:
            metrics = query2box.test_step(query2box, test_triples_ci, test_ans, test_ans_hard, args)
            log_metrics('Test ci', step, metrics)
            for metric in metrics:
                writer.add_scalar('Test_ci_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_ex_metrics[metric].append(metrics[metric])
        if 'ic' in tasks:
            metrics = query2box.test_step(query2box, test_triples_ic, test_ans, test_ans_hard, args)
            log_metrics('Test ic', step, metrics)
            for metric in metrics:
                writer.add_scalar('Test_ic_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_ex_metrics[metric].append(metrics[metric])
        if '2u' in tasks:
            metrics = query2box.test_step(query2box, test_triples_2u, test_ans, test_ans_hard, args)
            log_metrics('Test 2u', step, metrics)
            for metric in metrics:
                writer.add_scalar('Test_2u_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_u_metrics[metric].append(metrics[metric])
        if 'uc' in tasks:
            metrics = query2box.test_step(query2box, test_triples_uc, test_ans, test_ans_hard, args)
            log_metrics('Test uc', step, metrics)
            for metric in metrics:
                writer.add_scalar('Test_uc_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_u_metrics[metric].append(metrics[metric])
        for metric in average_metrics:
            writer.add_scalar('Test_average_'+metric, np.mean(average_metrics[metric]), step)
        for metric in average_c_metrics:
            writer.add_scalar('Test_average_c_'+metric, np.mean(average_c_metrics[metric]), step)
        for metric in average_c2_metrics:
            writer.add_scalar('Test_average_c2_'+metric, np.mean(average_c2_metrics[metric]), step)
        for metric in average_i_metrics:
            writer.add_scalar('Test_average_i_'+metric, np.mean(average_i_metrics[metric]), step)
        for metric in average_u_metrics:
            writer.add_scalar('Test_average_u_'+metric, np.mean(average_u_metrics[metric]), step)
        for metric in average_ex_metrics:
            writer.add_scalar('Test_average_ex_'+metric, np.mean(average_ex_metrics[metric]), step)

    def evaluate_val():
        average_metrics = collections.defaultdict(list)
        average_c_metrics = collections.defaultdict(list)
        average_c2_metrics = collections.defaultdict(list)
        average_i_metrics = collections.defaultdict(list)
        average_ex_metrics = collections.defaultdict(list)
        average_u_metrics = collections.defaultdict(list)
        if '2i' in tasks:
            metrics = query2box.test_step(query2box, valid_triples_2i, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 2i', step, metrics)
            for metric in metrics:
                writer.add_scalar('Valid_2i_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_i_metrics[metric].append(metrics[metric])
        if '3i' in tasks:
            metrics = query2box.test_step(query2box, valid_triples_3i, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 3i', step, metrics)
            for metric in metrics:
                writer.add_scalar('Valid_3i_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_i_metrics[metric].append(metrics[metric])
        if '2c' in tasks:
            metrics = query2box.test_step(query2box, valid_triples_2, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 2c', step, metrics)
            for metric in metrics:
                writer.add_scalar('Valid_2c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])
                average_c2_metrics[metric].append(metrics[metric])
        if '3c' in tasks:
            metrics = query2box.test_step(query2box, valid_triples_3, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 3c', step, metrics)
            for metric in metrics:
                writer.add_scalar('Valid_3c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])
                average_c2_metrics[metric].append(metrics[metric])
        if '1c' in tasks:
            metrics = query2box.test_step(query2box, valid_triples, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 1c', step, metrics)
            for metric in metrics:
                writer.add_scalar('Valid_1c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])
        if 'ci' in tasks:
            metrics = query2box.test_step(query2box, valid_triples_ci, valid_ans, valid_ans_hard, args)
            log_metrics('Valid ci', step, metrics)
            for metric in metrics:
                writer.add_scalar('Valid_ci_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_ex_metrics[metric].append(metrics[metric])
        if 'ic' in tasks:
            metrics = query2box.test_step(query2box, valid_triples_ic, valid_ans, valid_ans_hard, args)
            log_metrics('Valid ic', step, metrics)
            for metric in metrics:
                writer.add_scalar('Valid_ic_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_ex_metrics[metric].append(metrics[metric])
        if '2u' in tasks:
            metrics = query2box.test_step(query2box, valid_triples_2u, valid_ans, valid_ans_hard, args)
            log_metrics('Valid 2u', step, metrics)
            for metric in metrics:
                writer.add_scalar('Valid_2u_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_u_metrics[metric].append(metrics[metric])
        if 'uc' in tasks:
            metrics = query2box.test_step(query2box, valid_triples_uc, valid_ans, valid_ans_hard, args)
            log_metrics('Valid uc', step, metrics)
            for metric in metrics:
                writer.add_scalar('Valid_uc_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_u_metrics[metric].append(metrics[metric])
        for metric in average_metrics:
            writer.add_scalar('Valid_average_'+metric, np.mean(average_metrics[metric]), step)
        for metric in average_c_metrics:
            writer.add_scalar('Valid_average_c_'+metric, np.mean(average_c_metrics[metric]), step)
        for metric in average_c2_metrics:
            writer.add_scalar('Valid_average_c2_'+metric, np.mean(average_c2_metrics[metric]), step)
        for metric in average_i_metrics:
            writer.add_scalar('Valid_average_i_'+metric, np.mean(average_i_metrics[metric]), step)
        for metric in average_u_metrics:
            writer.add_scalar('Valid_average_u_'+metric, np.mean(average_u_metrics[metric]), step)
        for metric in average_ex_metrics:
            writer.add_scalar('Valid_average_ex_'+metric, np.mean(average_ex_metrics[metric]), step)
    
    def evaluate_train():
        average_metrics = collections.defaultdict(list)
        average_c_metrics = collections.defaultdict(list)
        average_c2_metrics = collections.defaultdict(list)
        average_i_metrics = collections.defaultdict(list)
        if '2i' in tasks:
            metrics = query2box.test_step(query2box, train_triples_2i, train_ans, train_ans, args)
            log_metrics('train 2i', step, metrics)
            for metric in metrics:
                writer.add_scalar('train_2i_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_i_metrics[metric].append(metrics[metric])
        if '3i' in tasks:
            metrics = query2box.test_step(query2box, train_triples_3i, train_ans, train_ans, args)
            log_metrics('train 3i', step, metrics)
            for metric in metrics:
                writer.add_scalar('train_3i_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_i_metrics[metric].append(metrics[metric])
        if '2c' in tasks:
            metrics = query2box.test_step(query2box, train_triples_2, train_ans, train_ans, args)
            log_metrics('train 2c', step, metrics)
            for metric in metrics:
                writer.add_scalar('train_2c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])
                average_c2_metrics[metric].append(metrics[metric])
        if '3c' in tasks:
            metrics = query2box.test_step(query2box, train_triples_3, train_ans, train_ans, args)
            log_metrics('train 3c', step, metrics)
            for metric in metrics:
                writer.add_scalar('train_3c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])
                average_c2_metrics[metric].append(metrics[metric])
        if '1c' in tasks:
            metrics = query2box.test_step(query2box, train_triples, train_ans, train_ans, args)
            log_metrics('train 1c', step, metrics)
            for metric in metrics:
                writer.add_scalar('train_1c_'+metric, metrics[metric], step)
                average_metrics[metric].append(metrics[metric])
                average_c_metrics[metric].append(metrics[metric])
        for metric in average_metrics:
            writer.add_scalar('train_average_'+metric, np.mean(average_metrics[metric]), step)
        for metric in average_c_metrics:
            writer.add_scalar('train_average_c_'+metric, np.mean(average_c_metrics[metric]), step)
        for metric in average_c2_metrics:
            writer.add_scalar('train_average_c2_'+metric, np.mean(average_c2_metrics[metric]), step)
        for metric in average_i_metrics:
            writer.add_scalar('train_average_i_'+metric, np.mean(average_i_metrics[metric]), step)

    if args.do_train:
        training_logs = []
        if args.task == '1c':
            begin_pq_step = args.max_steps
        else:
            begin_pq_step = args.max_steps - args.stepsforpath
        #Training Loop
        for step in range(init_step, args.max_steps):
            # print ("begining training step", step)
            # if step == 100:
            #     exit(-1)
            if step == 2*args.max_steps//3:
                args.valid_steps *= 4

            if step >= begin_pq_step and not args.train_onehop_only:
                if '2i' in tasks:
                    log = query2box.train_step(query2box, optimizer, train_iterator_2i, args, step)
                    for metric in log:
                        writer.add_scalar('2i_'+metric, log[metric], step)
                    training_logs.append(log)
                
                if '3i' in tasks:
                    log = query2box.train_step(query2box, optimizer, train_iterator_3i, args, step)
                    for metric in log:
                        writer.add_scalar('3i_'+metric, log[metric], step)
                    training_logs.append(log)
                
                if '2c' in tasks:
                    log = query2box.train_step(query2box, optimizer, train_iterator_2, args, step)
                    for metric in log:
                        writer.add_scalar('2c_'+metric, log[metric], step)
                    training_logs.append(log)
                
                if '3c' in tasks:
                    log = query2box.train_step(query2box, optimizer, train_iterator_3, args, step)
                    for metric in log:
                        writer.add_scalar('3c_'+metric, log[metric], step)
                    training_logs.append(log)

            if '1c' in tasks:
                log = query2box.train_step(query2box, optimizer, train_iterator, args, step)
                for metric in log:
                    writer.add_scalar('1c_'+metric, log[metric], step)
                training_logs.append(log)

            if training_logs == []:
                raise Exception("No tasks are trained!!")

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, query2box.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(query2box, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    if metric == 'inter_loss':
                        continue
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                inter_loss_sum = 0.
                inter_loss_num = 0.
                for log in training_logs:
                    if 'inter_loss' in log:
                        inter_loss_sum += log['inter_loss']
                        inter_loss_num += 1
                if inter_loss_num != 0:
                    metrics['inter_loss'] = inter_loss_sum / inter_loss_num
                log_metrics('Training average', step, metrics)
                training_logs = []
            
            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                evaluate_val()

        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(query2box, optimizer, save_variable_list, args)
        
    try:
        print (step)
    except:
        step = 0

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        evaluate_val()

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        evaluate_test()

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        evaluate_train()

    print ('Training finished!!')
    logging.info("training finished!!")


if __name__ == '__main__':
    main(parse_args())