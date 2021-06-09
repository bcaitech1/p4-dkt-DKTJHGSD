import multiprocessing
import os
import random
from collections import defaultdict
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import parmap
import torch
from sklearn.preprocessing import LabelEncoder


def convert_time(s):
    timestamp = datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple()
    return timestamp


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
    elif x < 4 * 60:
        return 'G'
    elif x < 24 * 60 * 60:
        return 'H'
    else:
        return 'I'


def process_by_userid(x, grouped, args):
    gp = grouped.get_group(int(x))
    gp = gp.sort_values(by=['userID', 'Timestamp'], ascending=True)

    if args.mode == 'pretrain':
        # Riiid 데이터에서 int 형태의 timestamp 사용
        gp['Timestamp'] = gp['Timestamp'].astype('int64')
        gp['time'] = gp['Timestamp'].shift(-1, fill_value=0)
        gp['time'] = gp['time'] - gp['Timestamp']

    else:
        tmp = gp['Timestamp'].astype(str)
        gp['Timestamp'] = tmp.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        gp['time'] = gp['Timestamp'].shift(-1, fill_value=datetime.strptime('1970-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'))
        gp['time'] = gp['time'] - gp['Timestamp']
        gp['time'] = gp['time'].apply(lambda x: int(x.total_seconds()))

        timetuple = tmp.apply(convert_time)
        gp['week_number'] = gp['Timestamp'].apply(lambda x: x.isocalendar()[1])  # 해당 년도의 몇번째 주인지
        gp['mday'] = timetuple.apply(lambda x: x.tm_wday)  # 요일
        gp['hour'] = timetuple.apply(lambda x: x.tm_hour)  # 시간

    gp['duration'] = gp['time'].apply(
        lambda x: x if x >= 0 else gp['time'][(gp['time'] <= 4 * 60) & (gp['time'] >= 0)].mean())
    gp['character'] = gp['time'].apply(get_character)

    # 문제 푼 수(전체, 태그별, 시험지별), 이동평균(전체, 태그별, 시험지별)  # 중첩 + window 포함
    record = defaultdict(list)
    c_record = defaultdict(int)
    gp['acc_tag_solved'], gp['acc_testid_solved'], gp['acc_tag_avg'], gp['acc_testid_avg'] = 0, 0, 0, 0
    gp['win_tag_solved'], gp['win_testid_solved'], gp['win_tag_avg'], gp['win_testid_avg'] = 0, 0, 0, 0
    filt = []
    for i in range(len(gp)):
        for type_ in ['acc', 'win']:
            for col in ['KnowledgeTag', 'testId']:
                record[type_ + col + 'avg'].append(float(round(
                    (c_record[type_ + str(gp[col].iloc[i]) + 'cor'] / c_record[type_ + str(gp[col].iloc[i])]) * 100,
                    2)) if c_record[type_ + str(gp[col].iloc[i])] != 0 else 0)
                record[type_ + col + 'solved'].append(c_record[type_ + str(gp[col].iloc[i])])
                c_record[type_ + str(gp[col].iloc[i])] += 1
                if gp['answerCode'].iloc[i] == 1:
                    c_record[type_ + str(gp[col].iloc[i]) + 'cor'] += 1
            if type_ == 'win':
                filt.append([gp['KnowledgeTag'].iloc[i], gp['testId'].iloc[i], gp['answerCode'].iloc[i]])
                if i >= args.max_seq_len:
                    tmp_know, tmp_testid, res = filt.pop(0)
                    c_record[type_ + str(tmp_know)] -= 1
                    c_record[type_ + str(tmp_testid)] -= 1
                    if res == 1:
                        c_record[type_ + str(tmp_know) + 'cor'] -= 1
                        c_record[type_ + str(tmp_testid) + 'cor'] -= 1

    gp['acc_tag_solved'], gp['acc_testid_solved'] = record['accKnowledgeTagsolved'], record['acctestIdsolved']
    gp['acc_tag_avg'], gp['acc_testid_avg'] = record['accKnowledgeTagavg'], record['acctestIdavg']
    gp['win_tag_solved'], gp['win_testid_solved'] = record['winKnowledgeTagsolved'], record['wintestIdsolved']
    gp['win_tag_avg'], gp['win_testid_avg'] = record['winKnowledgeTagavg'], record['wintestIdavg']

    return gp


def use_all(dt, max_seq_len, slide):
    seq_len = len(dt[0])
    tmp = np.stack(dt)
    new = [tuple([np.array(j) for j in tmp[:, i:i + max_seq_len]]) for i in range(0, seq_len - 8, max_seq_len // slide)]
    return new


def kfold_useall_data(train, val, args):
    # 모든 데이터 사용
    if args.by_window_or_by_testid == 'by_testid':
        data_1 = sum(
            parmap.map(partial(use_by_testid, max_seq_len=args.max_seq_len, test_cnt=args.testid_cnt, args=args),
                       train, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

        data_2 = sum(
            parmap.map(partial(use_by_testid, max_seq_len=args.max_seq_len, test_cnt=args.testid_cnt, args=args),
                       val, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])
    elif args.by_window_or_by_testid == 'by_window':
        data_1 = sum(parmap.map(partial(use_all, max_seq_len=args.max_seq_len, slide=args.slide_window),
                                train, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

        data_2 = sum(parmap.map(partial(use_all, max_seq_len=args.max_seq_len, slide=args.slide_window),
                                val, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])
    else:
        data_1_1 = sum(
            parmap.map(partial(use_by_testid, max_seq_len=args.max_seq_len, test_cnt=args.testid_cnt, args=args),
                       train, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

        data_2_1 = sum(
            parmap.map(partial(use_by_testid, max_seq_len=args.max_seq_len, test_cnt=args.testid_cnt, args=args),
                       val, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

        data_1_2 = sum(parmap.map(partial(use_all, max_seq_len=args.max_seq_len, slide=args.slide_window),
                                  train, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

        data_2_2 = sum(parmap.map(partial(use_all, max_seq_len=args.max_seq_len, slide=args.slide_window),
                                  val, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

        data_1 = data_1_1 + data_1_2
        data_2 = data_2_1 + data_2_2

    return data_1, data_2


def generate_mean_std(df, df_all, x):
    x_mean = df_all.groupby(x)['answerCode'].mean().reset_index()
    x_std = df_all.groupby(x)['answerCode'].std().reset_index()

    x_mean = {key: value for key, value in x_mean.values}
    x_std = {key: value for key, value in x_std.values}

    df_mean = df[x].apply(lambda x: x_mean[x])
    df_std = df[x].apply(lambda x: x_std[x])

    return df_mean, df_std


def use_by_testid(dt, max_seq_len, test_cnt, args, is_train=True):
    seq_len = len(dt[0])
    tmp = np.stack(dt)
    span = tmp[-1, :].astype(int)
    s = 0
    spans = []
    new = []
    if is_train:
        while s < seq_len:
            e = span[s]
            docs = []

            while e - s <= max_seq_len:
                docs.append((s, e))
                if e < seq_len:
                    e = span[e]
                else:
                    break

            for doc in docs[test_cnt - 1:]:
                spans.append(doc)
            s = span[s]

        for s, e in spans:
            if args.by_window_or_by_testid == 'both' and e - s == max_seq_len:
                continue
            new.append(tuple(np.array(j) for j in tmp[:-1, s:e]))
    else:
        if len(span) > max_seq_len:
            new.append(tuple(np.array(j) for j in tmp[:-1, span[-max_seq_len - 1]:]))
        else:
            new.append(tuple(np.array(j) for j in tmp[:-1, :]))
    return new


def make_max_min_idx(x, group):
    df = group.get_group(x).reset_index(drop=True)

    testid = df.loc[0, 'testId']
    index = []

    for idx in range(len(df)):
        if testid != df.loc[idx, 'testId']:
            df.loc[index, 'max_index'] = index[-1] + 1
            df.loc[index, 'min_index'] = index[0]
            index = [idx]
            testid = df.loc[idx, 'testId']
        else:
            index.append(idx)

    df.loc[index, 'max_index'] = index[-1] + 1
    df.loc[index, 'min_index'] = index[0]

    return df


class Preprocess:
    def __init__(self, args):
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
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        # 모든 데이터 사용
        if self.args.by_window_or_by_testid == 'by_testid':
            data_1 = sum(parmap.map(
                partial(use_by_testid, max_seq_len=self.args.max_seq_len, test_cnt=self.args.testid_cnt,
                        args=self.args),
                data_1, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

            data_2 = sum(parmap.map(
                partial(use_by_testid, max_seq_len=self.args.max_seq_len, test_cnt=self.args.testid_cnt,
                        args=self.args),
                data_2, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])
        elif self.args.by_window_or_by_testid == 'by_window':
            data_1 = sum(parmap.map(partial(use_all, max_seq_len=self.args.max_seq_len, slide=self.args.slide_window),
                                    data_1, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

            data_2 = sum(parmap.map(partial(use_all, max_seq_len=self.args.max_seq_len, slide=self.args.slide_window),
                                    data_2, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])
        else:
            data_1_1 = sum(parmap.map(
                partial(use_by_testid, max_seq_len=self.args.max_seq_len, test_cnt=self.args.testid_cnt,
                        args=self.args),
                data_1, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

            data_2_1 = sum(parmap.map(
                partial(use_by_testid, max_seq_len=self.args.max_seq_len, test_cnt=self.args.testid_cnt,
                        args=self.args),
                data_2, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

            data_1_2 = sum(parmap.map(partial(use_all, max_seq_len=self.args.max_seq_len, slide=self.args.slide_window),
                                      data_1, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

            data_2_2 = sum(parmap.map(partial(use_all, max_seq_len=self.args.max_seq_len, slide=self.args.slide_window),
                                      data_2, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])

            data_1 = data_1_1 + data_1_2
            data_2 = data_2_1 + data_2_2

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        if self.args.mode == 'pretrain':
            self.args.asset_dir = 'pretrain_' + self.args.asset_dir
        elif self.args.mode == 'inference':
            self.args.asset_dir = 'test_' + self.args.asset_dir

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        # df['max_index'] = 0
        # df['min_index'] = 0
        # df = parmap.map(partial(make_max_min_idx, group=df.groupby('userID')), df['userID'].unique(), pm_pbar=True,
        #                 pm_processes=multiprocessing.cpu_count())
        # df = pd.concat(df)

        for col in self.args.categorical_feats:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + '_classes.npy')
                le.classes_ = np.load(label_path)
                df[col] = df[col].apply(lambda x: str(x) if str(x) in le.classes_ else 'unknown')

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        return df

    def __feature_engineering(self, df):  # junho
        # 유져별로 feature engineering
        grouped = df.groupby(df.userID)
        final_df = sorted(list(df['userID'].unique()))
        final_df = parmap.map(partial(process_by_userid, grouped=grouped, args=self.args),
                              final_df, pm_pbar=True, pm_processes=multiprocessing.cpu_count())
        df = pd.concat(final_df)

        # mean, std
        if self.args.merge_train_test and self.args.mode != 'pretrain':
            df_train = pd.read_csv(os.path.join(self.args.data_dir, 'train_data.csv'))
            df_test = pd.read_csv(os.path.join(self.args.data_dir, 'test_data.csv'))
            df_test = df_test.loc[df.answerCode != -1]
            df_all = pd.concat([df_train, df_test])

        else:
            # pretrain의 경우 train/test dataset이 없으므로 전체 데이터를 그냥 사용
            df_all = df

        # difficulty mean, std
        df['difficulty'] = df['assessmentItemID'].apply(lambda x: x[1:4])
        df_all['difficulty'] = df_all['assessmentItemID'].apply(lambda x: x[1:4])
        df['difficulty_mean'], df['difficulty_std'] = generate_mean_std(df, df_all, 'difficulty')

        # assessmentItemID mean, std
        df['assId_mean'], df['assId_std'] = generate_mean_std(df, df_all, 'assessmentItemID')

        # tag mean, std
        df['tag_mean'], df['tag_std'] = generate_mean_std(df, df_all, 'KnowledgeTag')

        # testId mean, std
        df['testId_mean'], df['testId_std'] = generate_mean_std(df, df_all, 'testId')

        return df

    def load_data_from_file(self, file_name, is_train=True):

        processed_file_name_dict = {
            'pretrain': 'riiid_df.csv',
            'train': 'df.csv',
            'inference': 'test_df.csv'
        }

        save_file_path = os.path.join(self.args.output_dir, processed_file_name_dict[self.args.mode])
        if self.args.mode == 'train' and self.args.merge_train_test:
            save_file_path = os.path.join(self.args.output_dir, 'merged_df.csv')

        if os.path.isfile(save_file_path) and not self.args.reprocess_data:
            df = pd.read_csv(save_file_path)
        else:
            csv_file_path = os.path.join(self.args.data_dir, file_name)

            if self.args.mode == 'pretrain':
                df = pd.read_csv(csv_file_path)
            else:
                df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])  # , nrows=100000)

            df = self.__feature_engineering(df)
            df = self.__preprocessing(df, is_train)

            if self.args.mode == 'pretrain':
                df = df.fillna({'duration': df['duration'].mean(), 'assId_std': 0, 'testId_std': 0})

            df.to_csv(save_file_path, mode='w')  # dataframe csv파일로 저장


        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        cate_embeddings = defaultdict(int)
        if self.args.mode == 'pretrain':
            self.args.asset_dir = 'pretrain_' + self.args.asset_dir
        elif self.args.mode == 'inference':
            self.args.asset_dir = 'test_' + self.args.asset_dir

        for cate_name in self.args.categorical_feats:
            cate_embeddings[cate_name] = len(np.load(os.path.join(self.args.asset_dir, cate_name + '_classes.npy')))

        df = df.sort_values(by=['userID', 'Timestamp'], axis=0)
        columns = [i for i in list(df) if i != 'Timestamp']
        if self.args.by_window_or_by_testid == 'by_testid':
            val = sum(self.args.continuous_feats, []) + self.args.categorical_feats + ['answerCode'] + ['max_index']
        else:
            val = sum(self.args.continuous_feats, []) + self.args.categorical_feats + ['answerCode']

        group = df[columns].groupby('userID').apply(lambda r: tuple(r[i].values for i in val))

        return group.values, cate_embeddings

    def load_train_data(self, file_name):
        self.train_data, self.cate_embeddings = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data, self.cate_embeddings = self.load_data_from_file(file_name, is_train=False)
        if self.args.by_window_or_by_testid == 'by_testid':
            self.test_data = sum(parmap.map(
                partial(use_by_testid, max_seq_len=self.args.max_seq_len, test_cnt=self.args.testid_cnt, is_train=False,
                        args=self.args),
                self.test_data, pm_pbar=True, pm_processes=multiprocessing.cpu_count()), [])


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]
        # 각 data의 sequence length
        seq_len = len(row[0])
        feat_cols = list(row)

        max_seq_len = random.randint(10, self.args.max_seq_len) if self.args.to_random_seq else self.args.max_seq_len  # junho

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
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):
    pin_memory = True
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                                                   batch_size=args.batch_size, pin_memory=pin_memory,
                                                   collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                                                   batch_size=256, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader
