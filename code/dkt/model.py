import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import os
import math
import re

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel    

class LSTM(nn.Module):
    def __init__(self, args, cate_embeddings):
        super(LSTM, self).__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.bidirectional = args.bidirectional
        self.hd_div = args.hd_divider
        self.n_layers = self.args.n_layers
        self.num_feats = 1 + len(cate_embeddings) + len(self.args.continuous_feats)
        self.num_each_cont = [len(i) for i in self.args.continuous_feats]
        self.each_cont_idx = [[0, self.num_each_cont[0]]]

        for i in range(1, len(self.num_each_cont)):
            self.each_cont_idx.append([self.each_cont_idx[i-1][1], self.each_cont_idx[i-1][1] + self.num_each_cont[i]])

        # # 범주형 Embedding 
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.hd_div, padding_idx=0)
        self.embedding_cate = nn.ModuleList([nn.Embedding(cate_embeddings[i]+1, self.hidden_dim//self.hd_div, padding_idx = 0) 
                                                for i in cate_embeddings])

        # 연속형 Embedding
        self.embedding_cont = nn.ModuleList([nn.Sequential(nn.Linear(i, self.hidden_dim//self.hd_div), 
                                            nn.LayerNorm(self.hidden_dim//self.hd_div)) for i in self.num_each_cont])


        # # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//self.hd_div)*self.num_feats, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True,
                            bidirectional=self.args.bidirectional)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim * (2 if self.args.bidirectional else 1), 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        mask, interaction, _ = input[-3], input[-2], input[-1]
        cont_feats = input[:len(sum(self.args.continuous_feats,[]))]
        cate_feats = input[len(sum(self.args.continuous_feats,[])): -3]
        batch_size = interaction.size(0)

        # 범주형 Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_cate = [embed(cate_feats[idx]) for idx, embed in enumerate(self.embedding_cate)]

        # 연속형 Embedding
        cont_feats = [i.unsqueeze(2) for i in cont_feats]
        embed_cont = [embed(torch.cat(cont_feats[self.each_cont_idx[idx][0]:self.each_cont_idx[idx][1]],2)) for idx, embed in enumerate(self.embedding_cont)]
        
        embed = torch.cat([embed_interaction] + embed_cate + embed_cont, 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X)#, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim * (2 if self.args.bidirectional else 1))

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):
    def __init__(self, args, cate_embeddings):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.bidirectional = args.bidirectional
        self.hd_div = args.hd_divider
        self.num_feats = 1 + len(cate_embeddings) + len(self.args.continuous_feats)
        self.num_each_cont = [len(i) for i in self.args.continuous_feats]
        self.each_cont_idx = [[0, self.num_each_cont[0]]]

        for i in range(1, len(self.num_each_cont)):
            self.each_cont_idx.append([self.each_cont_idx[i-1][1], self.each_cont_idx[i-1][1] + self.num_each_cont[i]])

        # # 범주형 Embedding 
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.hd_div, padding_idx=0) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_cate = nn.ModuleList([nn.Embedding(cate_embeddings[i]+1, self.hidden_dim//self.hd_div, padding_idx = 0)for i in cate_embeddings])

        # 연속형 Embedding
        self.embedding_cont = nn.ModuleList([nn.Sequential(nn.Linear(i, self.hidden_dim//self.hd_div, bias=False),
                                            nn.LayerNorm(self.hidden_dim//self.hd_div)) for i in self.num_each_cont])


        # # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//self.hd_div)*self.num_feats, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True, 
                            bidirectional=self.bidirectional,
                            )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim  *(2 if self.bidirectional else 1),
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim *(2 if self.bidirectional else 1),
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim *(2 if self.bidirectional else 1), 1)

        self.activation = nn.Sigmoid()

        # T-Fixup
        if self.args.Tfixup:

            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixup Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0

        for name, param in self.named_parameters():
            if re.match(r'^embedding*', name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r'.*ln.*|.*bn.*', name):
                continue
            elif re.match(r'.*weight*', name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)

    def tfixup_scaling(self):
        temp_state_dict = {}

        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():

            # TODO: 모델 내부의 module 이름이 달라지면 직접 수정해서
            #       module이 scaling 될 수 있도록 변경해주자
            # print(name)

            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param
            elif re.match(r'encoder.*ffn.*weight$|encoder.*attn.out_proj.weight$', name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * param
            elif re.match(r"encoder.*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (param * (2**0.5))

        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]

        self.load_state_dict(temp_state_dict)

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):        
        mask, interaction, _ = input[-3], input[-2], input[-1]
        cont_feats = input[:len(sum(self.args.continuous_feats,[]))]
        cate_feats = input[len(sum(self.args.continuous_feats,[])): -3]
        batch_size = interaction.size(0)

        # 범주형 Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_cate = [embed(cate_feats[idx]) for idx, embed in enumerate(self.embedding_cate)]

        # 연속형 Embedding
        cont_feats = [i.unsqueeze(2) for i in cont_feats]
        embed_cont = [embed(torch.cat(cont_feats[self.each_cont_idx[idx][0]:self.each_cont_idx[idx][1]],2)) for idx, embed in enumerate(self.embedding_cont)]
        
        embed = torch.cat([embed_interaction] + embed_cate + embed_cont, 2)
        X = self.comb_proj(embed)

        # hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X)#, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim *(2 if self.bidirectional else 1))

        # extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        # encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        encoded_layers = self.attn(out, mask[:, None, :, :], head_mask=head_mask)
        print('encoded_layers: ', encoded_layers)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)
        preds = self.activation(out).view(batch_size, -1)
        return preds



class Bert(nn.Module):
    def __init__(self, args, cate_embeddings):
        super(Bert, self).__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.bidirectional = args.bidirectional
        self.hd_div = args.hd_divider
        self.num_feats = 1 + len(cate_embeddings) + len(self.args.continuous_feats)
        self.num_each_cont = [len(i) for i in self.args.continuous_feats]
        self.each_cont_idx = [[0, self.num_each_cont[0]]]

        for i in range(1, len(self.num_each_cont)):
            self.each_cont_idx.append([self.each_cont_idx[i-1][1], self.each_cont_idx[i-1][1] + self.num_each_cont[i]])

        # # 범주형 Embedding 
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.hd_div, padding_idx=0) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_cate = nn.ModuleList([nn.Embedding(cate_embeddings[i]+1, self.hidden_dim//self.hd_div, padding_idx = 0)for i in cate_embeddings])

        # 연속형 Embedding
        self.embedding_cont = nn.ModuleList([nn.Sequential(nn.Linear(i, self.hidden_dim//self.hd_div), 
                                            nn.LayerNorm(self.hidden_dim//self.hd_div)) for i in self.num_each_cont])


        # # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//self.hd_div)*self.num_feats, self.hidden_dim)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        mask, interaction, _ = input[-3], input[-2], input[-1]
        cont_feats = input[:len(sum(self.args.continuous_feats,[]))]
        cate_feats = input[len(sum(self.args.continuous_feats,[])): -3]
        batch_size = interaction.size(0)

        # 범주형 Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_cate = [embed(cate_feats[idx]) for idx, embed in enumerate(self.embedding_cate)]

        # 연속형 Embedding
        cont_feats = [i.unsqueeze(2) for i in cont_feats]
        embed_cont = [embed(torch.cat(cont_feats[self.each_cont_idx[idx][0]:self.each_cont_idx[idx][1]],2)) for idx, embed in enumerate(self.embedding_cont)]
    
        embed = torch.cat([embed_interaction] + embed_cate + embed_cont, 2)
        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds

class ConvBert(nn.Module): # chanhyeong
    def __init__(self, args, cate_embeddings):
        super(ConvBert, self).__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.bidirectional = args.bidirectional
        self.hd_div = args.hd_divider
        self.num_feats = 1 + len(cate_embeddings) + len(self.args.continuous_feats)
        self.num_each_cont = [len(i) for i in self.args.continuous_feats]
        self.each_cont_idx = [[0, self.num_each_cont[0]]]

        for i in range(1, len(self.num_each_cont)):
            self.each_cont_idx.append([self.each_cont_idx[i-1][1], self.each_cont_idx[i-1][1] + self.num_each_cont[i]])

        # # 범주형 Embedding 
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.hd_div, padding_idx=0) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_cate = nn.ModuleList([nn.Embedding(cate_embeddings[i]+1, self.hidden_dim//self.hd_div, padding_idx = 0)for i in cate_embeddings])

        # 연속형 Embedding
        self.embedding_cont = nn.ModuleList([nn.Sequential(nn.Linear(i, self.hidden_dim//self.hd_div), 
                                            nn.LayerNorm(self.hidden_dim//self.hd_div)) for i in self.num_each_cont])


        # # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//self.hd_div)*self.num_feats, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True,
                            bidirectional=self.args.bidirectional)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim * (2 if self.args.bidirectional else 1), 1)

        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len          
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)  
        self.conv1=nn.Conv1d(self.args.hidden_dim,self.args.hidden_dim,kernel_size=1)
        self.conv3=nn.Conv1d(self.args.hidden_dim,self.args.hidden_dim,kernel_size=3,padding=1)
        self.conv5=nn.Conv1d(self.args.hidden_dim,self.args.hidden_dim,kernel_size=5,padding=2)
        # Fully connected layer
        self.conv2fc = nn.Linear(self.args.hidden_dim*3,self.args.hidden_dim)
        self.fc = nn.Linear(self.args.hidden_dim, 1)
        self.activation = nn.Sigmoid()


    def forward(self, input):        
        mask, interaction, _ = input[-3], input[-2], input[-1]
        cont_feats = input[:len(sum(self.args.continuous_feats,[]))]
        cate_feats = input[len(sum(self.args.continuous_feats,[])): -3]
        batch_size = interaction.size(0)

       # 범주형 Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_cate = [embed(cate_feats[idx]) for idx, embed in enumerate(self.embedding_cate)]

        # 연속형 Embedding
        cont_feats = [i.unsqueeze(2) for i in cont_feats]
        embed_cont = [embed(torch.cat(cont_feats[self.each_cont_idx[idx][0]:self.each_cont_idx[idx][1]],2)) for idx, embed in enumerate(self.embedding_cont)]
    
        embed = torch.cat([embed_interaction] + embed_cate + embed_cont, 2)
        X = self.comb_proj(embed)

        # Bert
        
        #print("encoder input: ",X.shape)
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        #print("encoder 0th layer output: ",encoded_layers[0].shape)
        out2=out.clone()
        out2=torch.transpose(out2,1,2)
        #print("B,Cin,L: ",out2.shape)
        out2_conv1=self.conv1(out2)
        out2_conv3=self.conv3(out2.clone())
        out2_conv5=self.conv5(out2.clone())
        #print("conv1,conv3,conv5: ",out2_conv1.shape,out2_conv3.shape,out2_conv5.shape)
        concate_convs=torch.cat((out2_conv1, out2_conv3,out2_conv5), dim=1)
        #print("concated convs: ",concate_convs.shape)
        concate_convs = concate_convs.contiguous().view(batch_size, -1, self.hidden_dim*3)
        #print("processed concated convs: ",concate_convs.shape)
        output=self.conv2fc(concate_convs)
        #print("passed conv2fc: ",output.shape)
        output=self.fc(output)
        #print("passed final fc: ",output.shape)
#         out = out.contiguous().view(batch_size, -1, self.hidden_dim)
#         print("processed fc input: ",out.shape)        
#         out = self.fc(out)
#         print("fc output: ",out.shape)
#         preds = self.activation(out).view(batch_size, -1)
#         print("activation output=model output: ",preds,preds.shape)
        preds = self.activation(output).view(batch_size, -1)
        #print("activation output=model output: ",preds,preds.shape)

        return preds


# seoyoon
class LastQuery(nn.Module):  
    def __init__(self, args, cate_embeddings):
        super(LastQuery, self).__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.bidirectional = args.bidirectional
        self.hd_div = args.hd_divider
        self.num_feats = 1 + len(cate_embeddings) + len(self.args.continuous_feats)
        self.num_each_cont = [len(i) for i in self.args.continuous_feats]
        self.each_cont_idx = [[0, self.num_each_cont[0]]]


        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        for i in range(1, len(self.num_each_cont)):
            self.each_cont_idx.append([self.each_cont_idx[i-1][1], self.each_cont_idx[i-1][1] + self.num_each_cont[i]])

        # # 범주형 Embedding

        ## pretrained model과 dimension 맞추기 위해서 차원 변경
        if self.args.use_pretrained_model:
            cate_embeddings['testId'] = 61838
            cate_embeddings['assessmentItemID'] = 4934589

        # # 범주형 Embedding
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.hd_div, padding_idx=0) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_cate = nn.ModuleList([nn.Embedding(cate_embeddings[i]+1, self.hidden_dim//self.hd_div, padding_idx = 0) for i in cate_embeddings])

        # 연속형 Embedding
        self.embedding_cont = nn.ModuleList([nn.Sequential(nn.Linear(i, self.hidden_dim//self.hd_div, bias=False),
                                            nn.LayerNorm(self.hidden_dim//self.hd_div)) for i in self.num_each_cont])


        # # embedding combination projection
        if self.args.mode == 'pretrain':
            self.comb_proj_pre = nn.Linear((self.hidden_dim//self.hd_div)*self.num_feats, self.hidden_dim)
        else:
            self.comb_proj = nn.Linear((self.hidden_dim//self.hd_div)*self.num_feats, self.hidden_dim)


        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        #self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)

        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)

        if self.args.layer_norm:
            self.ln1 = nn.LayerNorm(self.hidden_dim)
            self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True,
                            bidirectional=self.args.bidirectional)


        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim * (2 if self.args.bidirectional else 1), 1)


        self.activation = nn.Sigmoid()

        if self.args.Tfixup:

            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixup Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0

        for name, param in self.named_parameters():
            if re.match(r'^embedding*', name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
                print('name2 : ', name)
            elif re.match(r'.*ln.*|.*bn.*', name):
                continue
            elif re.match(r'.*weight*', name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)


    def tfixup_scaling(self):
        temp_state_dict = {}

        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():

            # TODO: 모델 내부의 module 이름이 달라지면 직접 수정해서
            #       module이 scaling 될 수 있도록 변경해주자
            print(name)

            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param
            elif re.match(r'encoder.*ffn.*weight$|encoder.*attn.out_proj.weight$', name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * param
            elif re.match(r"encoder.*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (param * (2**0.5))

        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]

        self.load_state_dict(temp_state_dict)

    def mask_2d_to_3d(self, mask, batch_size, seq_len):
        # padding 부분에 1을 주기 위해 0과 1을 뒤집는다
        mask = torch.ones_like(mask) - mask

        mask = mask.repeat(1, seq_len)
        mask = mask.view(batch_size, -1, seq_len)
        mask = mask.repeat(1, self.args.n_heads, 1)
        mask = mask.view(batch_size*self.args.n_heads, -1, seq_len)

        return mask.masked_fill(mask==1, float('-inf'))

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)


    def forward(self, input):
        mask, interaction, _ = input[-3], input[-2], input[-1]
        cont_feats = input[:len(sum(self.args.continuous_feats,[]))]
        cate_feats = input[len(sum(self.args.continuous_feats,[])): -3]
        batch_size = interaction.size(0)

        # 범주형 Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_cate = [embed(cate_feats[idx]) for idx, embed in enumerate(self.embedding_cate)]

        # 연속형 Embedding
        cont_feats = [i.unsqueeze(2) for i in cont_feats]
        embed_cont = [embed(torch.cat(cont_feats[self.each_cont_idx[idx][0]:self.each_cont_idx[idx][1]],2)) for idx, embed in enumerate(self.embedding_cont)]

        embed = torch.cat([embed_interaction] + embed_cate + embed_cont, 2)

        if self.args.mode == 'pretrain':
            embed = self.comb_proj_pre(embed)
        else:
            embed = self.comb_proj(embed)

        # Positional Embedding
        #row = self.data[index]
        # 각 data의 sequence length
        #seq_len = len(row[0])
        # last query에서는 positional embedding을 하지 않음
        #position = self.get_pos(self.seq_len).to('cuda')
        #embed_pos = self.embedding_position(position)
        #embed = embed + embed_pos

        ####################### ENCODER #####################

        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        out, _ = self.attn(q, k, v, key_padding_mask=mask.squeeze())

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out

        if self.args.layer_norm:
            out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out

        if self.args.layer_norm:
            out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out) #, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim * (2 if self.args.bidirectional else 1))
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds

# seoyoon

class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))

class SAKTLSTM(nn.Module): #chanhyeong

    def __init__(self, args, cate_embeddings):
        super(SAKTLSTM, self).__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.norm1=nn.LayerNorm(self.args.hidden_dim)
        self.norm2=nn.LayerNorm(self.args.hidden_dim)
        self.norm3=nn.LayerNorm(self.args.hidden_dim)
        self.norm4=nn.LayerNorm(self.args.hidden_dim)
        self.norm5=nn.LayerNorm(self.args.hidden_dim)
        self.norm6=nn.LayerNorm(self.args.hidden_dim)
        self.norm7=nn.LayerNorm(self.args.hidden_dim)
        self.norm8=nn.LayerNorm(self.args.hidden_dim)
        self.norm9=nn.LayerNorm(self.args.hidden_dim)
        self.norm10=nn.LayerNorm(self.args.hidden_dim)

        self.drop_out = self.args.drop_out
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.bidirectional = args.bidirectional
        self.MLP_activ = F.leaky_relu
        self.hd_div = args.hd_divider
        self.num_feats = 1 + len(cate_embeddings) + len(self.args.continuous_feats)
        self.num_each_cont = [len(i) for i in self.args.continuous_feats]
        self.each_cont_idx = [[0, self.num_each_cont[0]]]
        self.dropout_layer=nn.Dropout(self.drop_out)


        for i in range(1, len(self.num_each_cont)):
            self.each_cont_idx.append([self.each_cont_idx[i-1][1], self.each_cont_idx[i-1][1] + self.num_each_cont[i]])

        # 범주형 embeiddng
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.hd_div, padding_idx=0)
        self.embedding_cate = nn.ModuleList([nn.Embedding(cate_embeddings[i]+1, self.hidden_dim//self.hd_div, padding_idx = 0)for i in cate_embeddings])

        # 연속형 Embedding
        self.embedding_cont = nn.ModuleList([nn.Sequential(nn.Linear(i, self.hidden_dim//self.hd_div),
                                            nn.LayerNorm(self.hidden_dim//self.hd_div)) for i in self.num_each_cont])



        # testid, assessmentitemID, knowledgetag, character, weeknumber, mday, hour
        # duration, difficulty_mean, assld_men, tage_mean, testid_mean
        # query : testid,assessmentitemId, knowledgetag, weeknumber, mday, hour 6
        # memory : duration,character, difficulty_mean, assid_mean, tag_mean, testid_mean 6
        self.linear1=nn.Linear((self.hidden_dim//self.hd_div)*6,self.hidden_dim*2)
        self.linear2=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.linear3=nn.Linear(self.hidden_dim+(self.hidden_dim//self.hd_div)*11,self.hidden_dim)
        self.linear4=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.linear5=nn.Linear((self.hidden_dim//self.hd_div)*6 + self.hidden_dim//2 , self.hidden_dim)
        self.linear6=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.linear7=nn.Linear(self.hidden_dim+(self.hidden_dim//self.hd_div)*11, self.hidden_dim)
        self.linear8=nn.Linear(self.hidden_dim,self.hidden_dim)


        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim//2,
                            self.n_layers,
                            batch_first=True
                            )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim*(2 if self.bidirectional else 1),
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim*(2 if self.bidirectional else 1),
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)
        self.attn2 = BertEncoder(self.config)
        self.attn3 = BertEncoder(self.config)

        self.attn.layer[0].attention.self.query=nn.Identity()
        self.attn.layer[0].attention.self.key=nn.Identity()
        self.attn.layer[0].attention.self.value=nn.Identity()
        self.attn2.layer[0].attention.self.query=nn.Identity()
        self.attn2.layer[0].attention.self.key=nn.Identity()
        self.attn2.layer[0].attention.self.value=nn.Identity()
        self.attn3.layer[0].attention.self.query=nn.Identity()
        self.attn3.layer[0].attention.self.key=nn.Identity()
        self.attn3.layer[0].attention.self.value=nn.Identity()


##### multiheadattention으로 encoder 구현############
#         self.mhattn=nn.MultiheadAttention(self.hidden_dim,self.n_heads,dropout=self.drop_out)
        self.mhattn_linear1=nn.Linear(self.hidden_dim,self.hidden_dim*2)
        self.mhattn_linear2=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.mhattn_linear3=nn.Linear(self.hidden_dim,self.hidden_dim*2)
        self.mhattn_linear4=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.mhattn_linear5=nn.Linear(self.hidden_dim,self.hidden_dim*2)
        self.mhattn_linear6=nn.Linear(self.hidden_dim*2,self.hidden_dim)
#######################################################

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim* (2 if self.bidirectional else 1), 1)
        self.activation = nn.Sigmoid()

        if self.args.Tfixup:

            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixup Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0

        for name, param in self.named_parameters():
            if re.match(r'^embedding_cate*', name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
                print('name2 : ', name)
            elif re.match(r'.*LayerNorm.*|.*norm.*|^embedding_cont.*.1.*', name):
                continue
            elif re.match(r'.*weight*', name):
                # nn.init.xavier_uniform_(param)
                print(name)
                nn.init.xavier_normal_(param)


    def tfixup_scaling(self):
        temp_state_dict = {}

        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():

            # TODO: 모델 내부의 module 이름이 달라지면 직접 수정해서
            #       module이 scaling 될 수 있도록 변경해주자
            print(name)

            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param
            elif re.match(r'.*LayerNorm.*|.*norm.*|^embedding_cont.*.1.*', name):
                continue
            elif re.match(r'attn*.*dense.*weight$|attn*.*attention.output.*weight$', name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * param
        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]

        self.load_state_dict(temp_state_dict)


    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        mask, interaction, _ = input[-3], input[-2], input[-1]
        # query_feats = category[0]testId,category[1]assessmentItemId,category[2]Tag,category[4]week,cate[5]mday,cate[6]hour 6개
        # memory_feats= cont[0]duration,cont[1]difficulty_mean,cont[2] diff_std, cont[3]assId_mean,cont[4] assid_std,cont[5]tag_mean,cont[6] tag_std,cont[7]testId_mean,cont[8] testid_std, cate[3]character, interaction 11
        cont_feats = input[:len(sum(self.args.continuous_feats,[]))]
        cate_feats = input[len(sum(self.args.continuous_feats,[])): -3]
        batch_size = interaction.size(0)

        # 범주형 Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_cate = [embed(cate_feats[idx]) for idx, embed in enumerate(self.embedding_cate)]

        # 연속형 Embedding
        cont_feats = [i.unsqueeze(2) for i in cont_feats]
        embed_cont = [embed(torch.cat(cont_feats[self.each_cont_idx[idx][0]:self.each_cont_idx[idx][1]],2)) for idx, embed in enumerate(self.embedding_cont)]

        #print(len(embed_cont),len(embed_cont[0][0][0]),len(embed_cont[1][0][0]))
        # embed_interaction, embed_cate, embed_cont
        raw_query_features = torch.cat([
                            embed_cate[0],
                            embed_cate[1],
                            embed_cate[2],
                            embed_cate[4],
                            embed_cate[5],
                            embed_cate[6]
                           ],2)
        memory_features = torch.cat([embed_cont[0],
                                     embed_cont[1],
                                     embed_cont[2],
                                     embed_cont[3],
                                     embed_cont[4],
                                     embed_cont[5],
                                     embed_cont[6],
                                     embed_cont[7],
                                     embed_cont[8],
                                     embed_cate[3],
                                     embed_interaction
                                    ],2)

        query_features = self.norm1(self.linear2(self.dropout_layer(self.MLP_activ(self.linear1(raw_query_features.clone())))))
        memory_cat = torch.cat([query_features.clone(),memory_features],2) # query_features = self.hidden_dim, + hiddn/div *7
        memory = self.norm2(self.linear4(self.dropout_layer(self.MLP_activ(self.linear3(memory_cat)))))

        lstm_out, hidden = self.lstm(memory)
        lstm_out = lstm_out.contiguous().view(batch_size,-1,self.hidden_dim//2)

        new_query = torch.cat([raw_query_features,lstm_out],2) # hidden_dim//div * 6 + hidden//2
        new_query_features = self.norm3(self.linear6(self.dropout_layer(self.MLP_activ(self.linear5(new_query)))))
        new_memory = torch.cat([new_query_features.clone(),memory_features],2) # self_hiddendim+div*7
        new_memory_features = self.norm4(self.linear8(self.dropout_layer(self.MLP_activ(self.linear7(new_memory)))))

        head_mask = [None] * self.n_heads


###### nn.multihead attention 으로 인코딩 레이어 구현하기
#         new_query_features=new_query_features.transpose(0,1)
#         new_memory_features=new_memory_features.transpose(0,1)


#         #print(padding_mask,padding_mask.shape)
#         # 내가 할일은, 여기서 -10000 -> 0, -0은 0으로.

#         # our mask consists of -10000. , -0. --> respecitvely, former is padded idx, latter is non-padded
#         #attention_triu_mask=torch.from_numpy(np.triu(np.ones((self.args.max_seq_len, self.args.max_seq_len)), k=1))
#         #attention_triu_mask=attention_triu_mask.masked_fill(attention_triu_mask == 1, float('-inf')).to(self.device)
#         #print(padding_mask,padding_mask.shape)
#         #print(attention_triu_mask,attention_triu_mask.shape)
#         encoded_output1=self.mhattn(new_query_features,
#                                     new_memory_features,
#                                     new_memory_features,
#                                     #attn_mask=attention_triu_mask,
#                                     key_padding_mask=mask)[0]
#         src = new_query_features+self.dropout_layer(encoded_output1)
#         src = self.norm(src)
#         src2 = self.mhattn_linear2(self.dropout_layer(self.MLP_activ(self.mhattn_linear1(src))))
#         src = src+self.dropout_layer(src2)
#         src = self.norm(src)
#         encoded_output2=self.mhattn(src,
#                                     new_memory_features,
#                                     new_memory_features,
#                                     #attn_mask=attention_triu_mask,
#                                     key_padding_mask=mask)[0]
#         src = new_query_features+self.dropout_layer(encoded_output2)
#         src = self.norm(src)
#         src2 = self.mhattn_linear2(self.dropout_layer(self.MLP_activ(self.mhattn_linear1(src))))
#         src = src+self.dropout_layer(src2)
#         src = self.norm(src)
#         sequence_output = src.reshape(-1,self.hidden_dim)


#####################

        encoded_1stlayers = self.attn(new_query_features,
                                   mask[:, None, :, :],
                                   head_mask=head_mask,
                                   encoder_hidden_states=new_memory_features,
                                   encoder_attention_mask=mask[:, None, :, :])

        src = new_query_features+self.dropout_layer(encoded_1stlayers[-1])
        src = self.norm5(src)
        src2 = self.mhattn_linear2(self.dropout_layer(self.MLP_activ(self.mhattn_linear1(src))))
        src = src+self.dropout_layer(src2)
        src = self.norm6(src)

        encoded_2ndlayers = self.attn2(src,
                                   mask[:, None, :, :],
                                   head_mask=head_mask,
                                   encoder_hidden_states=new_memory_features,
                                   encoder_attention_mask=mask[:, None, :, :])
        src = new_query_features+self.dropout_layer(encoded_2ndlayers[-1])
        src = self.norm7(src)
        src2 = self.mhattn_linear4(self.dropout_layer(self.MLP_activ(self.mhattn_linear3(src))))
        src = src+self.dropout_layer(src2)
        src = self.norm8(src)

        encoded_3rdlayers = self.attn3(src,
                                   mask[:, None, :, :],
                                   head_mask=head_mask,
                                   encoder_hidden_states=new_memory_features,
                                   encoder_attention_mask=mask[:, None, :, :])
        src = new_query_features+self.dropout_layer(encoded_3rdlayers[-1])
        src = self.norm9(src)
        src2 = self.mhattn_linear6(self.dropout_layer(self.MLP_activ(self.mhattn_linear5(src))))
        src = src+self.dropout_layer(src2)
        src = self.norm10(src)

        sequence_output = src.reshape(-1,self.hidden_dim)

#        sequence_output = encoded_2ndlayers[-1]
        out = self.fc(sequence_output)
        preds = self.activation(out).view(batch_size, -1)

        return preds

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


class Saint(nn.Module):

    def __init__(self, args, cate_embeddings):
        super(Saint, self).__init__()
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        self.hidden_dim = self.args.hidden_dim
        self.dropout = self.args.drop_out
        #self.dropout = 0.

        ##added parameters
        self.hd_div = args.hd_divider
        self.num_feats = 1 + len(cate_embeddings) + len(self.args.continuous_feats)
        self.num_each_cont = [len(i) for i in self.args.continuous_feats]
        self.each_cont_idx = [[0, self.num_each_cont[0]]]

        for i in range(1, len(self.num_each_cont)):
            self.each_cont_idx.append(
                [self.each_cont_idx[i - 1][1], self.each_cont_idx[i - 1][1] + self.num_each_cont[i]])

        ### Embedding
        # ENCODER embedding
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // self.hd_div)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // self.hd_div)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // self.hd_div)

        # encoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim // self.hd_div) * (self.num_feats+2), self.hidden_dim)
        #self.enc_comb_proj = nn.Linear((self.hidden_dim // self.hd_div) * self.num_feats , self.hidden_dim)

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)

        ## 범주형 Embedding
        #self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // self.hd_div,
                                                  padding_idx=0)  # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_cate = nn.ModuleList(
            [nn.Embedding(cate_embeddings[i] + 1, self.hidden_dim // self.hd_div, padding_idx=0) for i in
             cate_embeddings])

        # 연속형 Embedding
        self.embedding_cont = nn.ModuleList([nn.Sequential(nn.Linear(i, self.hidden_dim // self.hd_div),
                                                           nn.LayerNorm(self.hidden_dim // self.hd_div)) for i in
                                             self.num_each_cont])

        # decoder combination projection
        #self.dec_comb_proj = nn.Linear((self.hidden_dim // self.hd_div) * 4, self.hidden_dim)
        self.dec_comb_proj = nn.Linear((self.hidden_dim // self.hd_div) * (self.num_feats+3), self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers,
            num_decoder_layers=self.args.n_layers,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, input):
        # test, question, tag, _, mask, interaction, _ = input
        question = input[6]
        test = input[5]
        tag = input[7]
        #print(input[5], input[6], input[7])

        mask, interaction, _ = input[-3], input[-2], input[-1]
        cont_feats = input[:len(sum(self.args.continuous_feats, []))]
        cate_feats = input[len(sum(self.args.continuous_feats, [])): -3]

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        embed_interaction = self.embedding_interaction(interaction)
        embed_cate = [embed(cate_feats[idx]) for idx, embed in enumerate(self.embedding_cate)]
        cont_feats = [i.unsqueeze(2) for i in cont_feats]
        embed_cont = [embed(torch.cat(cont_feats[self.each_cont_idx[idx][0]:self.each_cont_idx[idx][1]], 2)) for
                      idx, embed in enumerate(self.embedding_cont)]

        # 신나는 embedding
        # ENCODER
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)


        embed_enc = torch.cat([embed_test,
                               embed_question,
                               embed_tag, ]
                              +embed_cate
                              +embed_cont, 2)

        #embed_enc = torch.cat([embed_interaction]
        #                      +embed_cate
        #                      +embed_cont, 2)

        embed_enc = self.enc_comb_proj(embed_enc)

        # DECODER
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)


        embed_dec = torch.cat([embed_test,
                               embed_question,
                               embed_tag,
                               embed_interaction]
                              +embed_cate
                              +embed_cont, 2)
        #embed_dec = torch.cat([embed_interaction]
        #                       +embed_cate
        #                       +embed_cont, 2)
        #print((self.hidden_dim // self.hd_div) * self.num_feats , embed_dec.shape)

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)

        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)

        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)

        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)

        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)

        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds


from .lana_model import LANA

def get_model(args, cate_embeddings): # junho
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args, cate_embeddings)
    elif args.model == 'lstmattn': model = LSTMATTN(args, cate_embeddings)
    elif args.model == 'bert': model = Bert(args, cate_embeddings)
    elif args.model == 'convbert': model= ConvBert(args, cate_embeddings) 
    elif args.model == 'lastquery': model= LastQuery(args, cate_embeddings)
    elif args.model == 'saint' : model = Saint(args, cate_embeddings)
    elif args.model == 'saktlstm': model=SAKTLSTM(args,cate_embeddings)
    elif args.model == 'lana' : model = LANA(args, cate_embeddings)
    return model


def load_model(args, file_name, cate_embeddings):
    model_path = os.path.join(args.model_dir, file_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args, cate_embeddings)

    # 1. load model state
    model.load_state_dict(load_state, strict=True)
   
    print("Loading Model from:", model_path, "...Finished")
    return model


class LastNQuery(LastQuery):
    def __init__(self, args, cate_embeddings):
        super(LastNQuery, self).__init__()

        self.query_agg = nn.Conv1d(in_channels=self.args.max_seq_len, out_channels=1, kernel_size=1)

    def forward(self, input):
        mask, interaction, _ = input[-3], input[-2], input[-1]
        cont_feats = input[:len(sum(self.args.continuous_feats, []))]
        cate_feats = input[len(sum(self.args.continuous_feats, [])): -3]
        batch_size = interaction.size(0)

        # 범주형 Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_cate = [embed(cate_feats[idx]) for idx, embed in enumerate(self.embedding_cate)]

        # 연속형 Embedding
        cont_feats = [i.unsqueeze(2) for i in cont_feats]
        embed_cont = [embed(torch.cat(cont_feats[self.each_cont_idx[idx][0]:self.each_cont_idx[idx][1]], 2)) for
                      idx, embed in enumerate(self.embedding_cont)]

        embed = torch.cat([embed_interaction] + embed_cate + embed_cont, 2)

        if self.args.mode == 'pretrain':
            embed = self.comb_proj_pre(embed)
        else:
            embed = self.comb_proj(embed)

        ####################### ENCODER #####################

        q = self.query_agg(self.query(embed)).permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        out, _ = self.attn(q, k, v, key_padding_mask=mask.squeeze())

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out

        if self.args.layer_norm:
            out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out

        if self.args.layer_norm:
            out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out)  # , hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim * (2 if self.args.bidirectional else 1))
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds
