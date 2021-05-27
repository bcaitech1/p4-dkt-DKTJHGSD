import os
import torch
import numpy as np
from tqdm.auto import tqdm

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
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
                # targets = input[3] # correct
                targets = batch[3]
                targets = targets.type(torch.FloatTensor)
                targets = targets.to(self.args.device)
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
                    # targets = input[3] # correct
                    targets = batch[3]
                    targets = targets.type(torch.FloatTensor)
                    targets = targets.to(self.args.device)
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

        test, question, tag, correct, duration, mask = batch

        # change to float
        mask = mask.type(torch.FloatTensor)
        correct = correct.type(torch.FloatTensor)

        #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
        #    saint의 경우 decoder에 들어가는 input이다
        # 패딩을 위해 correct값에 1을 더해준다. 0은 문제를 틀렸다라는 의미인데 우리는 0을 패딩으로 사용했기 때문에
        # 1을 틀림, 2를 맞음 으로 바꿔주는 작업. 아래 test, question, tag 같은 작업을 위해 모두 1을 더한다.
        interaction = correct + 1
        interaction = interaction.roll(shifts=1, dims=1)
        interaction[:, 0] = 0 # set padding index to the first sequence
        interaction = (interaction * mask).to(torch.int64)
        # print(interaction)
        # exit()
        #  test_id, question_id, tag
        test = ((test + 1) * mask).to(torch.int64)
        question = ((question + 1) * mask).to(torch.int64)
        tag = ((tag + 1) * mask).to(torch.int64)
        duration = ((duration + 1) * mask).to(torch.int64)

        # gather index
        # 마지막 sequence만 사용하기 위한 index
        gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
        gather_index = gather_index.view(-1, 1) - 1

        # device memory로 이동
        test = test.to(self.device)
        question = question.to(self.device)

        tag = tag.to(self.device)
        #    correct = correct.to(args.device)
        correct_adj = correct + 1
        correct_adj = correct_adj.to(self.args.device)
        mask = mask.to(self.device)

        interaction = interaction.to(self.device)
        gather_index = gather_index.to(self.device)
        duration = duration.to(self.device)

        return (test, question,
                tag, correct_adj, mask,
                interaction, gather_index, duration)


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
