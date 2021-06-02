import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='seed')
    #parser.add_argument('--device', default='cpu', type=str, help='cpu or gpu')

    parser.add_argument('--data_dir', default='/opt/ml/input/data/train_dataset', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='asset/', type=str, help='data directory')
    
    parser.add_argument('--file_name', default='train_data.csv', type=str, help='train file name')
    parser.add_argument('--model_dir', default='/opt/ml/p4-dkt-DKTJHGSD/code/models', type=str, help='model directory')

    parser.add_argument('--output_dir', default='output/', type=str, help='output directory')
    parser.add_argument('--test_file_name', default='test_data.csv', type=str, help='test file name')
    
    parser.add_argument('--max_seq_len', default=10, type=int, help='max sequence length')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')

    # 모델
    parser.add_argument('--hidden_dim', default=300, type=int, help='hidden dimension size')
    parser.add_argument('--hd_divider', default=15, type=int, help='hidden dimension divider')
    parser.add_argument('--n_layers', default=3, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=2, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.2, type=float, help='drop out rate')
    parser.add_argument('--bidirectional', default=True, type=bool, help='bi or uni directional')
    
    # 훈련
    parser.add_argument('--split_ratio', default=0.9, type=int, help='train val split ratio') # junho
    parser.add_argument('--n_epochs', default=40, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay',default=1e-3,type=float,help='weight decay') #changhyeong
    parser.add_argument('--clip_grad', default=10, type=int, help='clip grad')
    #parser.add_argument('--patience', default=5, type=int, help='for early stopping')
    parser.add_argument('--scheduler_gamma', default=0.5, type=float, help='lr decrease rate')
    parser.add_argument('--warmup_epoch', default=2, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=float, help = 'accumulating gradient') # junho
    parser.add_argument('--to_random_seq', default=False, type=bool, help = 'whether to use random max_seq') # junho
    parser.add_argument('--slide_window', default=10, type=int) # junho


    # feature
    parser.add_argument('--continuous_feats', type=list, nargs='+', 
            default=[['duration'], ['win_tag_solved','win_testid_solved'], ['win_tag_avg','win_testid_avg'], \
                    ], # ['acc_tag_solved','acc_testid_solved'], ['acc_tag_avg','acc_testid_avg']
            help = 'duration, acc_tag_solved, acc_tag_avg, acc_testid_solved, acc_testid_avg, \
                    win_tag_solved, win_tag_avg, win_testid_solved, win_testid_avg') # win == window, acc = accumulation

    parser.add_argument('--categorical_feats', type=list, nargs='+', 
            default=['testId', 'assessmentItemID', 'KnowledgeTag', 'character', 'difficulty', 'week_number', 'mday', 'hour'],
            help = 'testId, assessmentItemID, KnowledgeTag, character, difficulty, week_number, mday, hour')

    ## 중요 ##
    parser.add_argument('--model', default='lstmattn', type=str, help='model type')
    parser.add_argument('--optimizer', default='adamW', type=str, help='optimizer type')
    parser.add_argument('--scheduler', default='plateau', type=str, help='scheduler type') # [plateau, steplr, cosine, linear]
    parser.add_argument('--mode', default='train', type=str, help='train or inference') # junho
    parser.add_argument('--sweep',default=False, type=bool, help='if True, sweep mode') # chanhyeong
    parser.add_argument('--save_name',default="default",type=str,help='save name') # chanhyeong
    args = parser.parse_args()

    return args