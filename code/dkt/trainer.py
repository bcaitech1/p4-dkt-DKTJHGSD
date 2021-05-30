import os
import torch
import numpy as np
from tqdm.auto import tqdm

from .dataloader import get_loaders
from .criterion import get_criterion
from .metric import get_metric
from sklearn.metrics import accuracy_score

import wandb


class Trainer(object): # junho
    def __init__(self, args, model, epoch=None, optimizer=None, scheduler=None, train_dataset=None, test_dataset=None):
        self.args = args
        self.epoch = epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self):
        self.model.train()

        total_preds = []
        total_targets = []
        global_step, epoch_loss = 0, 0
        with tqdm(self.train_dataset, total = len(self.train_dataset), unit = 'batch') as train_bar:
            for step, batch in enumerate(train_bar):
                input = self.__process_batch(batch)
                preds = self.model(input)
                targets = input[-1] # correct
                loss = self.__compute_loss(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                epoch_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    global_step += 1
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.args.scheduler != 'plateau':
                        self.scheduler.step()  # Update learning rate schedule

                # if step % args.log_steps == 0:
                #     print(f"Training steps: {step} Loss: {str(loss.item())}")

                # predictions
                preds = preds[:,-1]
                targets = targets[:,-1]
                if str(self.device) == 'cuda:0':
                    preds = preds.to('cpu').detach().numpy()
                    targets = targets.to('cpu').detach().numpy()
                else: # cpu
                    preds = preds.detach().numpy()
                    targets = targets.detach().numpy()

                acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))
                current_lr = self.__get_lr(self.optimizer)

                total_preds.append(preds)
                total_targets.append(targets)

                ## update progress bar
                train_bar.set_description(f'Training [{self.epoch} / {self.args.n_epochs}]')
                train_bar.set_postfix(loss = loss.item(), acc = acc, current_lr = current_lr)

                wandb.log({"lr" : current_lr})

        total_preds = np.concatenate(total_preds)
        total_targets = np.concatenate(total_targets)

        # Train AUC / ACC
        auc, acc = get_metric(total_targets, total_preds)
        loss_avg = epoch_loss/global_step

        return auc, acc, loss_avg


    def validate(self):
        self.model.eval()

        total_preds = []
        total_targets = []
        eval_loss = 0
        with tqdm(self.test_dataset, total = len(self.test_dataset), unit = 'Evaluating') as eval_bar:
            with torch.no_grad():
                for step, batch in enumerate(eval_bar):
                    input = self.__process_batch(batch)
                    preds = self.model(input)
                    targets = input[-1] # correct
                    loss = self.__compute_loss(preds, targets)
                    # predictions
                    preds = preds[:,-1]
                    targets = targets[:,-1]

                    if str(self.device) == 'cuda:0':
                        preds = preds.to('cpu').detach().numpy()
                        targets = targets.to('cpu').detach().numpy()
                    else: # cpu
                        preds = preds.detach().numpy()
                        targets = targets.detach().numpy()
                    acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))

                    total_preds.append(preds)
                    total_targets.append(targets)

                    # 전체 손실 값 계산
                    eval_loss += loss.item()

                    # update progress bar
                    eval_bar.set_description(f'Evaluating [{self.epoch} / {self.args.n_epochs}]')
                    eval_bar.set_postfix(loss = loss.item(), acc = acc)

        total_preds = np.concatenate(total_preds)
        total_targets = np.concatenate(total_targets)

        # Train AUC / ACC
        auc, acc = get_metric(total_targets, total_preds)
        return auc, acc, eval_loss/len(self.test_dataset)


    def inference(self):
        self.model.eval()
        total_preds = []
        
        with tqdm(self.test_dataset, total = len(self.test_dataset), unit = 'Inference') as predict_bar:
            with torch.no_grad():
                for step, batch in enumerate(predict_bar):
                    input = self.__process_batch(batch)
                    preds = self.model(input)

                    # predictions
                    preds = preds[:,-1]

                    if str(self.device) == 'cuda:0':
                        preds = preds.to('cpu').detach().numpy()
                    else: # cpu
                        preds = preds.detach().numpy()
                    total_preds+=list(preds)

        write_path = os.path.join(self.args.output_dir, "output.csv")
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        with open(write_path, 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(write_path))
            w.write("id,prediction\n")
            for id, p in enumerate(total_preds):
                w.write('{},{}\n'.format(id,p))


    # 배치 전처리
    def __process_batch(self, batch):
        feats = batch[:-2]
        mask = batch[-1]
        correct = batch[-2]
        batch_size = mask.size()[0]
        # change to float
        mask = mask.type(torch.FloatTensor)
        correct = correct.type(torch.FloatTensor)

        #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
        #    saint의 경우 decoder에 들어가는 input이다
        # 패딩을 위해 correct값에 1을 더해준다. 0은 문제를 틀렸다라는 의미인데 우리는 0을 패딩으로 사용했기 때문에
        # 1을 틀림, 2를 맞음 으로 바꿔주는 작업. 아래 test, question, tag 같은 작업을 위해 모두 1을 더한다.
        interaction = correct + 1
        interaction = interaction.roll(shifts=1, dims=1)
        interaction_mask = mask.roll(shifts=1, dims=1)
        interaction_mask[:, 0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        
        trg_mask = torch.tril(torch.ones((self.args.max_seq_len, self.args.max_seq_len))).expand(
            batch_size, self.args.max_seq_len, self.args.max_seq_len
        )
        extended_mask = mask.unsqueeze(1)
        extended_mask = extended_mask# * trg_mask
        extended_mask = extended_mask.to(dtype=torch.float32)
        extended_mask = (1.0 - extended_mask) * -10000.0

        for i in range(len(feats)):
            filt = len(sum(self.args.continuous_feats,[]))
            if i >= filt:
                feats[i] = ((feats[i] + 1) * mask).to(torch.int64)

        # device memory로 이동
        for i in range(len(feats)):
            feats[i] = feats[i].to(self.device)

        interaction = interaction.to(self.device)
        correct = correct.to(self.device)
        mask = extended_mask.to(self.device)

        return feats + [mask, interaction, correct]


    # loss계산하고 parameter update!
    def __compute_loss(self, preds, targets):
        """
        Args :
            preds   : (batch_size, max_seq_len)
            targets : (batch_size, max_seq_len)

        """
        loss = get_criterion(preds, targets)
        #마지막 시퀀드에 대한 값만 loss 계산
        loss = loss[:,-1]
        loss = torch.mean(loss)
        return loss

    def __get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
