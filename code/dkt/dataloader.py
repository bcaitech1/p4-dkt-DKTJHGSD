import os
import time
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import random
from collections import defaultdict

import multiprocessing
from functools import partial
import parmap
from datetime import datetime

def get_character(x):
    if x < 0:
        return 'A'
    elif x < 23:
        return 'B'
    elif x < 56:
        return 'C'
    elif x < 68:
        return 'D'
    elif x < 84:
        return 'E'
    elif x < 108:
        return 'F'
    elif x < 4*60:
        return 'G'
    elif x < 24*60*60:
        return 'H'
    else :
        return 'I'

def convert_time(s):
    timestamp = datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple()
    return timestamp

def process_by_userid(x, grouped): # junho
    gp = grouped.get_group(int(x))
    gp = gp.sort_values(by=['userID','Timestamp'] ,ascending=True)
    tmp = gp['Timestamp'].astype(str)
    gp['Timestamp'] = tmp.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    gp['time'] = gp['Timestamp'].shift(-1, fill_value=datetime.strptime('1970-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'))
    gp['time'] = gp['time'] - gp['Timestamp']
    gp['time'] = gp['time'].apply(lambda x:int(x.total_seconds()))
    gp['duration'] = gp['time'].apply(lambda x: x if x >= 0 else gp['time'][(gp['time'] <= 4*60) & (gp['time'] >= 0)].mean())
    gp['character'] = gp['time'].apply(get_character)

    timetuple = tmp.apply(convert_time)
    gp['week_number'] = gp['Timestamp'].apply(lambda x:x.isocalendar()[1]) # 해당 년도의 몇번째 주인지
    gp['mday'] = timetuple.apply(lambda x:x.tm_wday) # 요일
    gp['hour'] = timetuple.apply(lambda x:x.tm_hour) # 시간
    
    # 문제 푼 수(전체, 태그별, 시험지별), 이동평균(전체, 태그별, 시험지별)
    record = defaultdict(int)
    gp['total_solved'], gp['tag_solved'], gp['testid_solved']  = 0, 0, 0
    gp['total_avg'], gp['tag_avg'], gp['testid_avg']  = 0, 0, 0
    total_avg, tag_avg, testid_avg, total_solved, tag_solved, testid_solved, flag = [], [], [], [], [], [], 0
    for i in range(len(gp)):
        if gp['time'].iloc[i] > 30*24*60*60:
            flag = 1
        elif flag == 1:
            record, flag = defaultdict(int), 0

        total_avg.append(float(round((record['total_correct']/record['total_solved'])*100, 2)) if record['total_solved'] != 0 else 0)
        tag_avg.append(float(round((record[str(gp['KnowledgeTag'].iloc[i])+'cor']/record[gp['KnowledgeTag'].iloc[i]])*100, 2)) if record[gp['KnowledgeTag'].iloc[i]] != 0 else 0)
        testid_avg.append(float(round((record[str(gp['testId'].iloc[i])+'cor']/record[gp['testId'].iloc[i]])*100,2)) if record[gp['testId'].iloc[i]] != 0 else 0)

        total_solved.append(record['total_solved'])
        record['total_solved'] += 1

        tag_solved.append(record[gp['KnowledgeTag'].iloc[i]])
        record[gp['KnowledgeTag'].iloc[i]] += 1

        testid_solved.append(record[gp['testId'].iloc[i]])
        record[gp['testId'].iloc[i]] += 1

        if gp['answerCode'].iloc[i] == 1:
            record['total_correct'] += 1
            record[str(gp['KnowledgeTag'].iloc[i])+'cor'] += 1
            record[str(gp['testId'].iloc[i])+'cor'] += 1

    gp['total_solved'], gp['tag_solved'], gp['testid_solved']  = total_solved, tag_solved, testid_solved
    gp['total_avg'], gp['tag_avg'], gp['testid_avg']  = total_avg, tag_avg, testid_avg
    return gp

def use_all(dt, max_seq_len, slide):
    seq_len = len(dt[0])
    tmp = np.stack(dt)
    new =[tuple([np.array(j) for j in tmp[:,i:i+max_seq_len]]) for i in range(0, seq_len, max_seq_len//slide)]
    # new=[]
    # for i in range(0, seq_len, max_seq_len//slide):
    #     check = tuple([np.array(j) for j in tmp[:,i:i+max_seq_len]])
    #     new.append(check)
    return new

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.cate_embeddings = None
        
    def get_train_data(self):
        return self.train_data, self.cate_embeddings

    def get_test_data(self):
        return self.test_data, self.cate_embeddings

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        # 모든 데이터 사용
        data_1 = sum(parmap.map(partial(use_all, max_seq_len = self.args.max_seq_len, slide=self.args.slide_window), 
                    data_1, pm_pbar = True, pm_processes = multiprocessing.cpu_count()), [])

        data_2 = sum(parmap.map(partial(use_all, max_seq_len = self.args.max_seq_len, slide=self.args.slide_window), 
                    data_2, pm_pbar = True, pm_processes = multiprocessing.cpu_count()), [])

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):
        # 수치형은 거른당 
        filt = ['userID','answerCode','Timestamp', 'time', 'total_solved', 'total_avg'] + sum(self.args.continuous_feats, [])
        cate_cols = [i for i in list(df) if i not in filt]
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:   
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                le.classes_ = np.load(label_path)
                
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
            
        #df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        return df

    def __feature_engineering(self, df): # junho   
        # 유져별로 feature engineering
        grouped = df.groupby(df.userID)
        final_df = sorted(list(df['userID'].unique()))
        final_df = parmap.map(partial(process_by_userid, grouped = grouped), 
                                      final_df, pm_pbar = True, pm_processes = multiprocessing.cpu_count())
        df = pd.concat(final_df)

        # 문제 난이도 추가
        test = pd.read_csv(os.path.join(self.args.data_dir, self.args.test_file_name)) 
        test['difficulty'] = test['assessmentItemID'].apply(lambda x:x[1:4])
        diff_rate = test.loc[test.answerCode!=-1].groupby('difficulty').mean().reset_index()
        diff_rate = diff_rate[['difficulty','answerCode']]
        diff_rate = {key:value for key, value in diff_rate.values}

        # df['difficulty'] = df['assessmentItemID'].apply(lambda x:x[1:4])
        # if self.args.mode =='inference':
        #     diff_rate = df.loc[df.answerCode!=-1].groupby('difficulty').mean().reset_index()
        # elif self.args.mode == 'train':
        #     diff_rate = df.groupby('difficulty').mean().reset_index()
        # diff_rate = diff_rate[['difficulty','answerCode']]
        # diff_rate = {key:value for key, value in diff_rate.values}

        df['difficulty'] = df['assessmentItemID'].apply(lambda x:x[1:4])
        df['difficulty'] = df['difficulty'].apply(lambda x: diff_rate[x])

        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name) #
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp']) #, nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)
        
        #df.to_csv('/opt/ml/p4-dkt-DKTJHGSD/code/output/df.csv', mode='w') # dataframe csv파일로 저장
        #df = pd.read_csv('/opt/ml/p4-dkt-DKTJHGSD/code/output/df.csv') # 저장한 dataframe 불러오기 

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        cate_embeddings = defaultdict(int)
        for cate_name in self.args.categorical_feats:
            cate_embeddings[cate_name] = len(np.load(os.path.join(self.args.asset_dir, cate_name + '_classes.npy')))

        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = [i for i in list(df) if i !='Timestamp']
        val = sum(self.args.continuous_feats, []) + self.args.categorical_feats + ['answerCode']
        group = df[columns].groupby('userID').apply(lambda r: tuple(r[i].values for i in val))

        return group.values, cate_embeddings

    def load_train_data(self, file_name):
        self.train_data, self.cate_embeddings = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data, self.cate_embeddings = self.load_data_from_file(file_name, is_train= False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]
        # 각 data의 sequence length
        seq_len = len(row[0])
        feat_cols = list(row)

        max_seq_len = random.randint(10, self.args.max_seq_len) if self.args.to_random_seq else self.args.max_seq_len #junho

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(feat_cols):
                feat_cols[i] = col[-max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        feat_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(feat_cols):
            feat_cols[i] = torch.FloatTensor(col) if i < len(sum(self.args.continuous_feats, [])) else torch.tensor(col)
        return feat_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

        
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] =torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):
    pin_memory = True # chanhyeong
    train_loader, valid_loader = None, None
    
    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=256, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader
