import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from training.gnn_mdi import train_gnn_mdi
from mc.mc_subparser import add_mc_subparser
from uci.uci_subparser import add_uci_subparser
from utils.utils import auto_select_gpu

def main(data, known, aggr, epochs, valid, log_dir, save_prediction):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None,) # default to be 1 hidden of node_dim
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
    parser.add_argument('--aggr', type=str, default='mean',)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight; 1: as input to mlp
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--impute_hiddens', type=str, default='64')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.7) # 1 - edge dropout rate
    parser.add_argument('--auto_known', action='store_true', default=False)
    parser.add_argument('--loss_mode', type=int, default = 0) # 0: loss on all train edge, 1: loss only on unknown train edge
    parser.add_argument('--valid', type=float, default=0.) # valid-set ratio
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='0')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_prediction', action='store_true', default=False)
    parser.add_argument('--transfer_dir', type=str, default=None)
    parser.add_argument('--transfer_extra', type=str, default='')
    parser.add_argument('--mode', type=str, default='train') # debug
    # subparsers = parser.add_subparsers()
    # add_uci_subparser(subparsers)
    # add_mc_subparser(subparsers)
    parser.add_argument('--domain', type=str, default='uci')
    parser.add_argument('--data', type=str, default='housing')
    parser.add_argument('--train_edge', type=float, default=0.7)
    parser.add_argument('--split_sample', type=float, default=0.)
    parser.add_argument('--split_by', type=str, default='y') # 'y', 'random'
    parser.add_argument('--split_train', action='store_true', default=False)
    parser.add_argument('--split_test', action='store_true', default=False)
    parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--node_mode', type=int, default=0)   # 0: feature onehot, sample all 1; 1: all onehot
    # args = parser.parse_args() 
    args, _ =parser.parse_known_args()

    data, known, aggr, epochs, valid, log_dir, save_prediction= 'energy', 0.9, 'mean', 2000, 0., 'task1_disc_ori', False
    # data, known, aggr, epochs, valid, log_dir, save_prediction= 'Breast', 0.9, 'mean', 2000, 0., 'task4_2', False
    args.data = data
    args.known = known
    args.aggr = aggr
    args.epochs = epochs
    args.valid = valid
    args.log_dir = log_dir
    args.save_prediction = save_prediction
    args.task = 'original'
    # args.task = '1-1'

    # task 1
    args.data_type = 'reg'
    args.continuous = [0, 1, 2, 3, 4, 6]
    args.discrete = [5, 7]
    args.node_mode = 0
    if args.node_mode == 4:
        args.impute_ce = True
    else:
        args.impute_ce = False

    # task 2
    args.impute_loss_weight = 0.5
    args.predict_loss_weight = 0.5

    # task 3 (Few-shot)
    args.data_type = 'cls'
    args.train_edge = 0.7
    args.train_y = 0.7
    args.known = 0.9

    # task 3 (Unsupervised)
    args.task = '3-2'
    print(args)

    # task 4
    args.data_type = 'cls'
    args.task = '4'
    args.temperature = 0.07

    # select device
    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        device = torch.device('cuda:{}'.format(cuda))
    else:
        print('Using CPU')
        device = torch.device('cpu')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.domain == 'uci':
        from uci.uci_data import load_data
        data = load_data(args)
    elif args.domain == 'mc':
        from mc.mc_data import load_data
        data = load_data(args)
    print(data)

    log_path = './{}/test/{}/{}/'.format(args.domain, args.data, args.log_dir)
    if os.path.isdir(log_path) == False:
        os.makedirs(log_path)

    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)

    train_gnn_mdi(data, args, log_path, device)


if __name__ == '__main__':
    main()