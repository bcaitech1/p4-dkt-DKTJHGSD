import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=71, type=int, help='seed')

    parser.add_argument('--data_dir', default='/opt/ml/input/data/train_dataset', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='asset/', type=str, help='data directory')

    parser.add_argument('--file_name', default='train_data.csv', type=str, help='train file name')
    parser.add_argument('--model_dir', default='models/', type=str, help='model directory')

    parser.add_argument('--output_dir', default='output/', type=str, help='output directory')
    parser.add_argument('--test_file_name', default='test_data.csv', type=str, help='test file name')

    parser.add_argument('--max_seq_len', default=30, type=int, help='max sequence length')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')

    # 모델
    parser.add_argument('--hidden_dim', default=300, type=int, help='hidden dimension size')
    parser.add_argument('--hd_divider', default=15, type=int, help='hidden dimension divider')
    parser.add_argument('--n_layers', default=2, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=4, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.2, type=float, help='drop out rate')
    parser.add_argument('--bidirectional', default=True, type=bool, help='bi or uni directional')

    # 훈련
    parser.add_argument('--split_ratio', default=0.9, type=int, help='train val split ratio') 
    parser.add_argument('--kfold', default=0, type=int, help='utilize kfold') 
    parser.add_argument('--n_epochs', default=40, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')  
    parser.add_argument('--clip_grad', default=10, type=int, help='clip grad')
    parser.add_argument('--patience', default=6, type=int, help='for early stopping')
    parser.add_argument('--scheduler_gamma', default=0.5, type=float, help='lr decrease rate')
    parser.add_argument('--warmup_epoch', default=4, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=float, help = 'accumulating gradient') 
    parser.add_argument('--to_random_seq', default=False, type=bool, help = 'whether to use random max_seq')
    parser.add_argument('--slide_window', default=20, type=int) 
    parser.add_argument('--by_window_or_by_testid', default='by_testid', type=str, help='choose split data method')
    parser.add_argument('--testid_cnt', default=3, type=int, help='minimum testid_cnt, 0 choose by length')
    parser.add_argument('--Tfixup', default=False, type=bool, help='Utilize Tfixup')
    parser.add_argument('--layer_norm', default=True, type=bool, help='Utilize layer_norm')
    parser.add_argument('--pseudo_labeling', default=4, type=int, help='# times to replace answer w/ pseudo labeling')


    # feature
    parser.add_argument('--continuous_feats', type=list, nargs='+',
                        default=[['duration'], ['difficulty_mean', 'difficulty_std'], ['assId_mean'], \
                                ['tag_mean', 'tag_std'], ['testId_mean', 'testId_std']],
            help = 'duration, tag_solved, tag_avg, testid_solved, testid_avg, difficulty_mean, \
                    difficulty_sum, difficulty_std, assId_mean, assId_sum, assId_std,tag_mean, \
                    tag_sum, tag_std, testId_mean, testId_sum, testId_std, acc_tag_solved, \
                    acc_tag_avg, acc_testid_solved, acc_testid_avg, win_tag_solved, win_tag_avg,\
                    win_tag_solved, win_tag_avg, win_testid_solved, win_testid_avg, lag_time')

    parser.add_argument('--categorical_feats', type=list, nargs='+',
                        default=['testId', 'assessmentItemID', 'character', 'KnowledgeTag', 'week_number', 'mday', 'hour'],
                        help='testId, assessmentItemID, KnowledgeTag, character, week_number, mday, hour')

    ## 중요 ##
    parser.add_argument('--model', default='lstmattn', type=str, 
                        help='model type: lstm, lstmattn, bert, convbert, lastquery, saint, saktlstm, lana')
    parser.add_argument('--optimizer', default='adamW', type=str, help='optimizer type')
    parser.add_argument('--scheduler', default='plateau', type=str,
                        help='scheduler type : plateau, steplr, cosine, linear')
    parser.add_argument('--mode', default='train', type=str, help='pretrain, train, pseudo_labeling or inference')
    parser.add_argument('--use_pretrained_model', default=False, type=bool,
                        help='if True, use pretrained model when training a model')
    parser.add_argument('--reprocess_data', default=False, type=bool,
                        help='if True, reprocess data using feature engineering and preprocessing')
    parser.add_argument('--sweep', default=False, type=bool, help='if True, sweep mode')
    parser.add_argument('--save_name', default="default", type=str, help='save name')
    args = parser.parse_args()

    return args
