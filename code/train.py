import os
import shutil
from args import parse_args
from dkt.dataloader import Preprocess
from dkt.engine import run
from dkt.utils import setSeeds, get_timestamp
import wandb

hyperparameter_defaults = dict(
batch_size = 64,
learning_rate = 0.001,
weight_decay=0.01
) #chanhyeong, sweep config 껍데기

def main(args):
    setSeeds(args.seed)
    preprocess = Preprocess(args)

    if args.mode == 'train': #junho
        wandb.login()
        preprocess.load_train_data(args.file_name)
        train_data, cate_embeddings = preprocess.get_train_data()
        train_data, valid_data = preprocess.split_data(train_data, ratio=args.split_ratio, seed=args.seed)  
        name = '(' + args.model + ')' + ' ' + get_timestamp()
        if args.sweep : #chanhyeong
            wandb.init(project="sweep", config=hyperparameter_defaults)
            sweep_cfg = wandb.config
            args.batch_size=sweep_cfg.batch_size
            args.lr=sweep_cfg.learning_rate
            args.weight_decay=sweep_cfg.weight_decay
        else:
            wandb.init(project='dkt', config=vars(args), name = name)
        run(args, train_data = train_data, valid_data = valid_data, cate_embeddings = cate_embeddings)
        shutil.rmtree('/opt/ml/p4-dkt-DKTJHGSD/code/wandb') # 완드비 폴더 삭제 

    elif args.mode =='inference': # junho
        preprocess.load_test_data(args.test_file_name)
        test_data, cate_embeddings = preprocess.get_test_data()
        run(args, test_data = test_data, cate_embeddings = cate_embeddings)
    

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True) 
    main(args)