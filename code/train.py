import os
import shutil
from args import parse_args
from dkt.dataloader import Preprocess
from dkt.engine import run
from dkt.utils import setSeeds, get_timestamp
import wandb


def main(args):
    setSeeds(args.seed)
    preprocess = Preprocess(args)

    if args.mode == 'train': # junho
        wandb.login()
        preprocess.load_train_data(args.file_name)
        train_data = preprocess.get_train_data()
        train_data, valid_data = preprocess.split_data(train_data, ratio=args.split_ratio, seed=args.seed)   
        name = '(' + args.model + ')' + ' ' + get_timestamp()
        wandb.init(project='dkt', config=vars(args), entity='jlee621', name = name)
        run(args, train_data = train_data, valid_data = valid_data)
        shutil.rmtree('/opt/ml/p4-dkt-DKTJHGSD/code/wandb') # 완드비 폴더 삭제 

    elif args.mode =='inference': # junho
        preprocess.load_test_data(args.test_file_name)
        test_data = preprocess.get_test_data()
        run(args, test_data = test_data)
    

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True) 
    main(args)