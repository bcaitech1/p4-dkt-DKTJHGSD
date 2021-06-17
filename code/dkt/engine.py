import torch
import os
from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import call_scheduler
from .model import get_model, load_model  # junho
from .trainer import Trainer
import wandb
import math
import copy
import pandas as pd


def run(args, train_data = None, valid_data = None, test_data = None, cate_embeddings = None, fold = None, pseudo_cnt = None, pseudo_mode=None):
    if args.mode == 'train' or args.mode == 'pretrain':
        train_loader, valid_loader = get_loaders(args, train_data, valid_data)

        # only when using warmup scheduler
        args.total_steps = math.ceil(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
        args.one_step = math.ceil(len(train_loader.dataset) / args.batch_size) 

        model = get_model(args, cate_embeddings)

        if args.use_pretrained_model:
            model_path = os.path.join(args.model_dir, 'pretrain.pt')
            model.load_state_dict(torch.load(model_path), strict=False)  # 이어서 학습
            print("===============Pretrained model is loaded.=======================")
        optimizer = get_optimizer(model, args)
        scheduler = call_scheduler(optimizer, args)

        best_auc, best_epoch = -1, 1
        if args.kfold:
            print(f'Training fold : {fold}')
            early_stopping_counter = 0
        for epoch in range(args.n_epochs):
            ### TRAIN
            trainer = Trainer(args, model, epoch + 1, optimizer, scheduler, train_loader, valid_loader)
            train_auc, train_acc, train_loss = trainer.train()

            ### VALID
            eval_auc, eval_acc, eval_loss = trainer.validate()

            print(
                f'\tTrain Loss: {train_loss:.3f} | Train Acc: {round(train_acc * 100, 2)}% | Train AUC: {round(train_auc * 100, 2)}%')
            print(
                f'\tValid Loss: {eval_loss:.3f} | Valid Acc: {round(eval_acc * 100, 2)}% | Valid AUC: {round(eval_auc * 100, 2)}%')

            if args.kfold: 
                wandb.log({f"k{fold}_valid_auc":eval_auc})
            else:
                wandb.log({"train_loss": train_loss, "train_auc": train_auc, "train_acc": train_acc,
                           "valid_loss": eval_loss, "valid_auc": eval_auc, "valid_acc": eval_acc})

            if eval_auc > best_auc:
                best_auc, best_epoch = eval_auc, epoch + 1
                # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                model_to_save = model.module if hasattr(model, 'module') else model
                if not os.path.exists(args.model_dir):
                    os.makedirs(args.model_dir)
                if args.kfold:
                    torch.save(model_to_save.state_dict(),
                               os.path.join(args.model_dir, f'{args.save_name}_{fold}.pt')) 
                else:
                    torch.save(model_to_save.state_dict(), os.path.join(args.model_dir, f'{args.save_name}.pt'))
                print('\tbetter model found, saving!')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                    return best_auc

            # scheduler
            if args.scheduler == 'plateau':
                scheduler.step(best_auc)
        print('=' * 50 + f' Training finished, best model found in epoch : {best_epoch} ' + '=' * 50)
        return best_auc
    
    elif (args.pseudo_labeling > 0) & (pseudo_mode=='labeling') :
        _, test_loader = get_loaders(args, None, test_data)
        model = load_model(args, f'{args.save_name}_pseudo{pseudo_cnt-1}.pt', cate_embeddings)
        make_label = Trainer(args, model, test_dataset = test_loader) 
        test_predict = make_label.inference()
        print('[  test_predict  ] : ', len(test_predict))
        pseudo_labels=[]
        for x in test_predict:
            pseudo_labels.append(1 if x>=0.5 else 0)
        print('[  pseudo_labels  ] : ', len(pseudo_labels))

        temp_test_data = pd.read_csv('/opt/ml/input/data/train_dataset/pseudo_df_test.csv', parse_dates=['Timestamp']) # 저장한 dataframe 불러오기
        temp_test_data = temp_test_data.sort_values(by=['userID','Timestamp'], axis=0)
            
        # pseudo 라벨이 담길 test 데이터 복사본
        pseudo_test_data = copy.deepcopy(temp_test_data)

        # pseudo label 테스트 데이터 update
        for i, pseudo_label in enumerate(pseudo_labels):
            pseudo_test_data.answerCode[i] = pseudo_label

        pseudo_df_train = pd.read_csv('/opt/ml/input/data/train_dataset/pseudo_df_train.csv', parse_dates=['Timestamp']) # 저장한 dataframe 불러오기
        pseudo_df_train['pseudo'] = False
        pseudo_test_data['pseudo'] = True
        pseudo_train_data = pd.concat([pseudo_df_train, pseudo_test_data])
        
        print(pseudo_train_data.head())
        print(f"train 셋 크기       : {len(pseudo_df_train)}")
        print(f"test 셋 크기        : {len(pseudo_test_data)}")
        print("-" * 30)
        print(f"새로운 train 셋 크기 : {len(pseudo_train_data)}")

        return pseudo_train_data

    elif args.mode == 'inference':
        print("Start Inference")
        _, test_loader = get_loaders(args, None, test_data)
        if args.kfold:
            model = load_model(args, f'{args.save_name}_{fold}.pt', cate_embeddings)
        else:
            model = load_model(args, f'{args.save_name}.pt', cate_embeddings) 
        inference = Trainer(args, model, test_dataset=test_loader, fold=fold)
        inference.inference()
        print('=' * 50 + ' Inference finished ' + '=' * 50)
