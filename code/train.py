import os
import shutil
from time import sleep
from args import parse_args
from dkt.dataloader import Preprocess, kfold_useall_data
from dkt.engine import run
from dkt.utils import setSeeds, get_timestamp, kfold_ensemble
import wandb
from sklearn.model_selection import KFold
import time

hyperparameter_defaults = dict(
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01
)  # sweep config 껍데기


def main(args):
    setSeeds(args.seed)
    name = '(' + args.model + ')' + ' ' + get_timestamp()
    preprocess = Preprocess(args)

    if args.mode == 'train' or args.mode == 'pretrain': 
        wandb.login()
        start_time = time.time() # 시작 시간 기록
        preprocess.load_train_data(args.file_name)
        train_data, cate_embeddings = preprocess.get_train_data()
        if args.kfold:
            wandb.init(project='dkt', config=vars(args), name=name)
            kf, cnt, accu_auc, best_fold, best_auc = KFold(n_splits=args.kfold), 1, 0, 0, 0
            for train_idx, val_idx in kf.split(train_data):
                train, valid = train_data[train_idx], train_data[val_idx]
                train, valid = kfold_useall_data(train, valid, args)
                auc = run(args, train_data=train, valid_data=valid, cate_embeddings=cate_embeddings, fold=cnt)
                accu_auc += auc
                if auc > best_auc:
                    best_auc, best_fold = auc, cnt
                cnt += 1 
            print(f'Average AUC : {round(accu_auc/args.kfold,2)}')
            print(f'Best_fold : {best_fold} | Best AUC : {best_auc}')
        else:
            train_data, valid_data = preprocess.split_data(train_data, ratio=args.split_ratio, seed=args.seed)
            
            end_time = time.time() # 종료 시간 기록
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
            print(f'Sliding window : {args.slide_window}')
            print(f'Time Spent on data preprocessing : {elapsed_mins} minutes {elapsed_secs} seconds')
            
            if args.sweep:  
                wandb.init(project="sweep", config=hyperparameter_defaults)
                sweep_cfg = wandb.config
                args.batch_size = sweep_cfg.batch_size
                args.lr = sweep_cfg.learning_rate
                args.weight_decay = sweep_cfg.weight_decay
            else:
                wandb.init(project='dkt', config=vars(args), name = name)
            run(args, train_data = train_data, valid_data = valid_data, cate_embeddings = cate_embeddings)
        
        #shutil.rmtree('/opt/ml/p4-dkt-DKTJHGSD/code/wandb') # 완드비 폴더 삭제 

    elif args.mode =='pseudo_labeling':
        if args.pseudo_labeling :
            #train
            print("="*30 , "Start pseudo labeling", "="*30)
            print("-"*20, "Start train #1")
            preprocess.load_train_data(args.file_name, preprocessed_file=False)    
            train_data, cate_embeddings = preprocess.get_train_data()
            train_data, valid_data = preprocess.split_data(train_data, ratio=args.split_ratio, seed=args.seed)
            name = name+'pseudo1'
            wandb.init(project='dkt', config=vars(args), name = name)
            run(args, train_data = train_data, valid_data = valid_data, cate_embeddings = cate_embeddings, pseudo_cnt=1, pseudo_mode=None)
            print("-"*20, "End train #1")
            for i in range(1, args.pseudo_labeling):
                print("="*30 , "Start pseudo labeling #", i+1, "="*30)
                
                #학습된 model로 labeled dataset만들기
                preprocess.load_test_data(args.test_file_name)
                test_data, cate_embeddings = preprocess.get_test_data()
                pseudo_train_data = run(args, test_data = test_data, cate_embeddings = cate_embeddings, pseudo_cnt=i+1, pseudo_mode='labeling')
                pseudo_train_data.to_csv(f'/opt/ml/input/data/train_dataset/pseudo_labeling.csv', mode='w') # dataframe csv파일로 저장
                
                #만들어진 labeled dataset 로 train
                print("-"*20, "Start train #", i+1)
                preprocess.load_train_data(f'pseudo_labeling.csv')
                train_data, cate_embeddings = preprocess.get_train_data()
                train_data, valid_data = preprocess.split_data(train_data, ratio=args.split_ratio, seed=args.seed)          
                name = name+'pseudo'+ (i+1) 
                wandb.init(project='dkt', config=vars(args), name = name)
                run(args, train_data = train_data, valid_data = valid_data, cate_embeddings = cate_embeddings, pseudo_cnt=i+1, pseudo_mode=None)
                print("-"*20, "End train #", i+1)
                print("="*30 , "End pseudo labeling #", i+1, "="*30)

    elif args.mode == 'inference':  
        preprocess.load_test_data(args.test_file_name)
        test_data, cate_embeddings = preprocess.get_test_data()
        if args.kfold:
            for i in range(1, args.kfold + 1):
                run(args, test_data=test_data, cate_embeddings=cate_embeddings, fold=i)
            kfold_ensemble(os.path.join(args.output_dir, "kfold_outputs"), args.output_dir)
        else:
            run(args, test_data=test_data, cate_embeddings=cate_embeddings)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
