import os
import shutil
from time import sleep
from args import parse_args
from dkt.dataloader import Preprocess, kfold_useall_data
from dkt.engine import run
from dkt.utils import setSeeds, get_timestamp, kfold_ensemble
import wandb
from sklearn.model_selection import KFold

hyperparameter_defaults = dict(
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01
)  # chanhyeong, sweep config 껍데기


def main(args):
    setSeeds(args.seed)
    name = '(' + args.model + ')' + ' ' + get_timestamp()
    preprocess = Preprocess(args)

    if args.mode == 'train' or args.mode == 'pretrain':  # junho
        wandb.login()
        preprocess.load_train_data(args.file_name)
        train_data, cate_embeddings = preprocess.get_train_data()
        if args.kfold:
<<<<<<< HEAD
            wandb.init(project='DKT', config=vars(args), name = name)
=======
            wandb.init(project='dkt', config=vars(args), name=name)
>>>>>>> c900c12ca22d1daac3d09db909d2ce074f311518
            kf, cnt, accu_auc, best_fold, best_auc = KFold(n_splits=args.kfold), 1, 0, 0, 0
            for train_idx, val_idx in kf.split(train_data):
                train, valid = train_data[train_idx], train_data[val_idx]
                train, valid = kfold_useall_data(train, valid, args)
                auc = run(args, train_data=train, valid_data=valid, cate_embeddings=cate_embeddings, fold=cnt)
                accu_auc += auc
                if auc > best_auc:
                    best_auc, best_fold = auc, cnt
<<<<<<< HEAD
                cnt += 1 
            print(f'Average AUC : {round(accu_auc/args.kfold,2)}')
            print(f'Best_fold : {best_fold} | Best AUC : {best_auc}')
=======
                cnt += 1
            print(f'Average AUC : {round(accu_auc / args.kfold, 2)}')
            print(f'Best_fold / AUC : {best_fold} / {best_auc}')
>>>>>>> c900c12ca22d1daac3d09db909d2ce074f311518
        else:
            train_data, valid_data = preprocess.split_data(train_data, ratio=args.split_ratio, seed=args.seed)
            if args.sweep:  # chanhyeong
                wandb.init(project="sweep", config=hyperparameter_defaults)
                sweep_cfg = wandb.config
                args.batch_size = sweep_cfg.batch_size
                args.lr = sweep_cfg.learning_rate
                args.weight_decay = sweep_cfg.weight_decay
            else:
<<<<<<< HEAD
                wandb.init(project='DKT', config=vars(args), name = name)
            run(args, train_data = train_data, valid_data = valid_data, cate_embeddings = cate_embeddings)
        
        #shutil.rmtree('/opt/ml/p4-dkt-DKTJHGSD/code/wandb') # 완드비 폴더 삭제 
=======
                wandb.init(project='dkt', config=vars(args), name=name)
            run(args, train_data=train_data, valid_data=valid_data, cate_embeddings=cate_embeddings)

        # shutil.rmtree('/opt/ml/p4-dkt-DKTJHGSD/code/wandb') # 완드비 폴더 삭제
>>>>>>> c900c12ca22d1daac3d09db909d2ce074f311518

    elif args.mode == 'inference':  # junho
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
