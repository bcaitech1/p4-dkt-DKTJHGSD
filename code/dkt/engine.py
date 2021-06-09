import torch
import os
from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import call_scheduler
from .model import get_model, load_model # junho
from .trainer import Trainer
import wandb
import math

def run(args, train_data = None, valid_data = None, test_data = None, cate_embeddings = None, fold = None):
    if args.mode == 'train' or args.mode == 'pretrain':
        train_loader, valid_loader = get_loaders(args, train_data, valid_data)
        
        # only when using warmup scheduler
        args.total_steps = math.ceil(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
        args.one_step = math.ceil(len(train_loader.dataset) / args.batch_size) #junho
        #args.warmup_steps = args.total_steps // 10 
                
        model = get_model(args, cate_embeddings)

        if args.use_pretrained_model:
            model.load_state_dict(torch.load('/opt/ml/p4-dkt-DKTJHGSD/code/models/default.pt'), strict=False) # 이어서 학습
        optimizer = get_optimizer(model, args)
        scheduler = call_scheduler(optimizer, args)

        best_auc, best_epoch = -1, 1
        if args.kfold:
            print(f'Training fold : {fold}')
            early_stopping_counter = 0
        for epoch in range(args.n_epochs):
            ### TRAIN
            trainer = Trainer(args, model, epoch+1, optimizer, scheduler, train_loader, valid_loader) # junho
            train_auc, train_acc, train_loss = trainer.train()
        
            ### VALID
            eval_auc, eval_acc, eval_loss = trainer.validate()

            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {round(train_acc*100, 2)}% | Train AUC: {round(train_auc*100, 2)}%')
            print(f'\tValid Loss: {eval_loss:.3f} | Valid Acc: {round(eval_acc*100, 2)}% | Valid AUC: {round(eval_auc*100, 2)}%')

            if args.kfold: 
                wandb.log({f"k{fold}_train_loss": train_loss, f"k{fold}_train_auc": train_auc, f"k{fold}_train_acc":train_acc,
                        f"k{fold}_valid_loss" : eval_loss, f"k{fold}_valid_auc":eval_auc, f"k{fold}_valid_acc":eval_acc})
            else:
                wandb.log({"train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                        "valid_loss" : eval_loss, "valid_auc":eval_auc, "valid_acc":eval_acc})

            if eval_auc > best_auc:
                best_auc, best_epoch = eval_auc, epoch+1
                # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                model_to_save = model.module if hasattr(model, 'module') else model
                if not os.path.exists(args.model_dir):
                    os.makedirs(args.model_dir)
                if args.kfold:    
                    torch.save(model_to_save.state_dict(), os.path.join(args.model_dir, f'{args.save_name}_{fold}.pt')) #chanhyeong
                else: torch.save(model_to_save.state_dict(), os.path.join(args.model_dir, f'{args.save_name}.pt'))
                print('\tbetter model found, saving!')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                    break

            # scheduler
            if args.scheduler == 'plateau':
                scheduler.step(best_auc)
        print('='*50 + f' Training finished, best model found in epoch : {best_epoch} ' + '='*50)
    
    elif args.mode == 'inference':
        print("Start Inference")
        _, test_loader = get_loaders(args, None, test_data)
        if args.kfold:
            model = load_model(args, f'{args.save_name}_{fold}.pt', cate_embeddings)
        else: 
            model = load_model(args, f'{args.save_name}.pt', cate_embeddings) #chanhyeong
        inference = Trainer(args, model, test_dataset = test_loader, fold = fold) # junho
        inference.inference()
        print('='*50 + ' Inference finished ' + '='*50)

