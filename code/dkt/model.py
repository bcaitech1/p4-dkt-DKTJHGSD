import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import os
import math
import re
import copy

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel    


###############################################################################################
####################################   LSTM   #################################################
###############################################################################################
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



###############################################################################################
####################################   LSTM Attention   #######################################
###############################################################################################
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
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, mask[:, None, :, :], head_mask=head_mask)
        print('encoded_layers: ', encoded_layers)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)
        preds = self.activation(out).view(batch_size, -1)
        return preds


###############################################################################################
####################################   BERT   #################################################
###############################################################################################
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



###############################################################################################
####################################   ConvBert   #############################################
###############################################################################################
class ConvBert(nn.Module):
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
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out2=out.clone()
        out2=torch.transpose(out2,1,2)
        out2_conv1=self.conv1(out2)
        out2_conv3=self.conv3(out2.clone())
        out2_conv5=self.conv5(out2.clone())
        concate_convs=torch.cat((out2_conv1, out2_conv3,out2_conv5), dim=1)
        concate_convs = concate_convs.contiguous().view(batch_size, -1, self.hidden_dim*3)
        output=self.conv2fc(concate_convs)
        output=self.fc(output)
        preds = self.activation(output).view(batch_size, -1)

        return preds



###############################################################################################
####################################   LastQuery  #############################################
###############################################################################################
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
        for i in range(1, len(self.num_each_cont)):
            self.each_cont_idx.append([self.each_cont_idx[i-1][1], self.each_cont_idx[i-1][1] + self.num_each_cont[i]])

        # # 범주형 Embedding
        ## pretrained model과 dimension 맞추기 위해서 차원 변경
        if self.args.use_pretrained_model:
            cate_embeddings['testId'] = 61838
            cate_embeddings['assessmentItemID'] = 4934589

        # # 범주형 Embedding
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.hd_div, padding_idx=0) 
        self.embedding_cate = nn.ModuleList([nn.Embedding(cate_embeddings[i]+1, self.hidden_dim//self.hd_div, padding_idx = 0) 
                                                            for i in cate_embeddings])

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




###############################################################################################
####################################   SAKTLSTM   #############################################
###############################################################################################
class SAKTLSTM(nn.Module):
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
        self.embedding_cate = nn.ModuleList([nn.Embedding(cate_embeddings[i]+1, self.hidden_dim//self.hd_div, padding_idx = 0)
                                                            for i in cate_embeddings])

        # 연속형 Embedding
        self.embedding_cont = nn.ModuleList([nn.Sequential(nn.Linear(i, self.hidden_dim//self.hd_div),
                                            nn.LayerNorm(self.hidden_dim//self.hd_div)) for i in self.num_each_cont])

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
        self.mhattn_linear1=nn.Linear(self.hidden_dim,self.hidden_dim*2)
        self.mhattn_linear2=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.mhattn_linear3=nn.Linear(self.hidden_dim,self.hidden_dim*2)
        self.mhattn_linear4=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.mhattn_linear5=nn.Linear(self.hidden_dim,self.hidden_dim*2)
        self.mhattn_linear6=nn.Linear(self.hidden_dim*2,self.hidden_dim)

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
        cont_feats = input[:len(sum(self.args.continuous_feats,[]))]
        cate_feats = input[len(sum(self.args.continuous_feats,[])): -3]
        batch_size = interaction.size(0)

        # 범주형 Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_cate = [embed(cate_feats[idx]) for idx, embed in enumerate(self.embedding_cate)]

        # 연속형 Embedding
        cont_feats = [i.unsqueeze(2) for i in cont_feats]
        embed_cont = [embed(torch.cat(cont_feats[self.each_cont_idx[idx][0]:self.each_cont_idx[idx][1]],2)) for idx, embed in enumerate(self.embedding_cont)]

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
        out = self.fc(sequence_output)
        preds = self.activation(out).view(batch_size, -1)

        return preds


###############################################################################################
####################################  Saint  ##################################################
###############################################################################################
class Saint(nn.Module):
    def __init__(self, args, cate_embeddings):
        super(Saint, self).__init__()
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = self.args.hidden_dim
        self.dropout = self.args.drop_out
        self.hd_div = args.hd_divider
        self.num_feats = 1 + len(cate_embeddings) + len(self.args.continuous_feats)
        self.num_each_cont = [len(i) for i in self.args.continuous_feats]
        self.each_cont_idx = [[0, self.num_each_cont[0]]]

        for i in range(1, len(self.num_each_cont)):
            self.each_cont_idx.append(
                [self.each_cont_idx[i - 1][1], self.each_cont_idx[i - 1][1] + self.num_each_cont[i]])

        ## ENCODER ##
        # ENCODER embedding
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // self.hd_div)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // self.hd_div)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // self.hd_div)

        # encoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim // self.hd_div) * (self.num_feats+2), self.hidden_dim)

        # categorical Embedding
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // self.hd_div,
                                                  padding_idx=0)  # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_cate = nn.ModuleList(
            [nn.Embedding(cate_embeddings[i] + 1, self.hidden_dim // self.hd_div, padding_idx=0) for i in
             cate_embeddings])

        # continuous Embedding
        self.embedding_cont = nn.ModuleList([nn.Sequential(nn.Linear(i, self.hidden_dim // self.hd_div),
                                                           nn.LayerNorm(self.hidden_dim // self.hd_div)) for i in
                                             self.num_each_cont])

        ## DECODER ##
        # decoder combination projection
        self.dec_comb_proj = nn.Linear((self.hidden_dim // self.hd_div) * (self.num_feats+3), self.hidden_dim) #interaction + cate_

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

        # T-Fixup
        if self.args.Tfixup:
            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixup Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

    def tfixup_initialization(self):
        padding_idx = 0

        for name, param in self.named_parameters():
            if len(param.shape) == 1:# bypass bias parameters
                continue
            if re.match(r'^embedding*', name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r'.*Norm.*', name):
                continue
            elif re.match(r'.*weight*', name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)

    def tfixup_scaling(self):
        temp_state_dict = {}
        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():
            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param
            elif re.match(r'.*Norm.*', name):
                continue
            elif re.match(r'encoder.*dense.*weight$|encoder.*attention.output.*weight$', name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * param
            elif re.match(r"encoder.*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (param * (2 ** 0.5))

        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]

        self.load_state_dict(temp_state_dict)

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, input):
        question = input[10]# assessmentItemID
        test = input[9]# testId
        tag = input[11]#KnowledgeTag

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

        ## ENCODER ##
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        # exercise information
        embed_enc = torch.cat([embed_test,
                               embed_question,
                               embed_tag, ]
                              + embed_cate
                              + embed_cont, 2)
        embed_enc = self.enc_comb_proj(embed_enc)


        ## DECODER ##
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        # response
        embed_dec = torch.cat([embed_test,
                               embed_question,
                               embed_tag,
                               embed_interaction]
                              + embed_cate
                              + embed_cont, 2)

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK 생성
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



###############################################################################################
####################################   LastNQuery   ###########################################
###############################################################################################
class LastNQuery(LastQuery):
    def __init__(self, args, cate_embeddings):
        super(LastNQuery, self).__init__(args, cate_embeddings)

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




###############################################################################################
####################################   LANA    ################################################
###############################################################################################
class LANA(nn.Module):
    def __init__(self, args, cate_embeddings, n_encoder=1, n_decoder=1):
        super(LANA, self).__init__()
        self.args = args
        self.max_seq = self.args.max_seq_len
        self.c_emb = cate_embeddings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.hd_div = args.hd_divider

        self.pos_embed = PositionalBias(self.args, self.max_seq, self.args.hidden_dim, self.args.n_heads, bidirectional=False, num_buckets=32)


        ## encoder
        self.encoder_interaction = nn.Embedding(3, self.args.hidden_dim//self.hd_div, padding_idx=0) 
        self.encoder_testid = nn.Embedding(cate_embeddings['testId']+1, self.args.hidden_dim//self.hd_div, padding_idx=0)  
        self.encoder_assid = nn.Embedding(cate_embeddings['assessmentItemID']+1, self.args.hidden_dim//self.hd_div, padding_idx=0)                                
        self.encoder_tag = nn.Embedding(cate_embeddings['KnowledgeTag']+1,self.args.hidden_dim//self.hd_div, padding_idx=0)
        self.encoder_dif_ms = nn.Sequential(nn.Linear(2, self.args.hidden_dim//self.hd_div), nn.LeakyReLU(),
                                              nn.LayerNorm(self.args.hidden_dim//self.hd_div))
        self.encoder_ass_m = nn.Sequential(nn.Linear(1, self.args.hidden_dim//self.hd_div),  nn.LeakyReLU(),
                                              nn.LayerNorm(self.args.hidden_dim//self.hd_div))
        self.encoder_testid_ms = nn.Sequential(nn.Linear(2, self.args.hidden_dim//self.hd_div),  nn.LeakyReLU(),
                                              nn.LayerNorm(self.args.hidden_dim//self.hd_div))
        self.encoder_tag_ms = nn.Sequential(nn.Linear(2, self.args.hidden_dim//self.hd_div),  nn.LeakyReLU(),
                                              nn.LayerNorm(self.args.hidden_dim//self.hd_div))

        self.encoder_linear = nn.Linear(8 * (self.args.hidden_dim//self.hd_div),  self.args.hidden_dim)
        self.encoder_layernorm = nn.LayerNorm(self.args.hidden_dim)
        self.encoder_dropout = nn.Dropout(self.args.drop_out)

        ## decoder
        self.decoder_duration = nn.Sequential(nn.Linear(1, self.args.hidden_dim//self.hd_div), nn.LeakyReLU(),
                                              nn.LayerNorm(self.args.hidden_dim//self.hd_div)) 
        self.decoder_lagtime = nn.Sequential(nn.Linear(1, self.args.hidden_dim//self.hd_div), nn.LeakyReLU(),
                                              nn.LayerNorm(self.args.hidden_dim//self.hd_div))
        self.decoder_wnum = nn.Embedding(cate_embeddings['week_number']+1, self.args.hidden_dim//self.hd_div, padding_idx=0) 
        self.decoder_mday = nn.Embedding(cate_embeddings['mday']+1, self.args.hidden_dim//self.hd_div, padding_idx=0) 
        self.decoder_hour = nn.Embedding(cate_embeddings['hour']+1, self.args.hidden_dim//self.hd_div, padding_idx=0) 
        self.decoder_character = nn.Embedding(cate_embeddings['character']+1, self.args.hidden_dim//self.hd_div, padding_idx=0) 
        
        self.decoder_linear = nn.Linear(6 * (self.args.hidden_dim//self.hd_div), self.args.hidden_dim)
        self.decoder_layernorm = nn.LayerNorm(self.args.hidden_dim)
        self.decoder_dropout = nn.Dropout(self.args.drop_out)

        self.encoder = get_clones(LANAEncoder(self.args, self.args.hidden_dim, self.args.n_heads, self.args.hidden_dim, 
                                                self.max_seq), n_encoder)
        self.srfe = BaseSRFE(self.args, self.args.hidden_dim, self.args.n_heads)
        self.decoder = get_clones(LANADecoder(self.args, self.args.hidden_dim, self.args.n_heads, self.args.hidden_dim, 
                                                self.max_seq), n_decoder)

        self.layernorm_out = nn.LayerNorm(self.args.hidden_dim)
        self.ffn = PivotFFN(self.args.hidden_dim, self.args.hidden_dim, 32, self.args.drop_out)
        self.classifier = nn.Linear(self.args.hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def get_pos_seq(self):
        return torch.arange(self.max_seq).unsqueeze(0)

    def forward(self, input):
        testid = input[9]
        assid = input[10]
        part = input[12]
        tag = input[11]
        character, week_num, mday, hour = input[13], input[14], input[15], input[16]
        
        dif_mean, dif_std, ass_mean, testid_mean, testid_std, tag_mean, tag_std = input[1:8]
        dif_mean, dif_std, ass_mean = dif_mean.unsqueeze(2), dif_std.unsqueeze(2), ass_mean.unsqueeze(2)
        testid_mean, testid_std, tag_mean, tag_std = testid_mean.unsqueeze(2), testid_std.unsqueeze(2), tag_mean.unsqueeze(2), tag_std.unsqueeze(2)

        duration = input[0]
        lagtime = input[1]
        interaction = input[-2]
        batch_size = interaction.size(0)


        ltime = lagtime.clone()
        pos_embed = self.pos_embed(self.get_pos_seq().to(self.device))

        # encoder embedding
        testid_seq = self.encoder_testid(testid) # content id
        assid_seq = self.encoder_assid(assid) 
        part_seq = self.encoder_part(part) # part
        tag_seq = self.encoder_tag(tag)
        interaction_seq = self.encoder_interaction(interaction)
        dif_ms_seq = self.encoder_dif_ms(torch.cat([dif_mean, dif_std], 2))
        ass_m_seq = self.encoder_ass_m(ass_mean)
        testid_ms_seq = self.encoder_testid_ms(torch.cat([testid_mean, testid_std], 2))
        tag_ms_seq = self.encoder_tag_ms(torch.cat([tag_mean, tag_std], 2))

        encoder_input = torch.cat([interaction_seq, part_seq, assid_seq, testid_seq, tag_seq, dif_ms_seq, ass_m_seq, testid_ms_seq, tag_ms_seq], dim=-1)
        encoder_input = self.encoder_linear(encoder_input)
        encoder_input = self.encoder_layernorm(encoder_input)
        encoder_input = self.encoder_dropout(encoder_input)

        # decoder embedding
        #d_correct_seq = self.decoder_resp_embed(correct) # correctness
        duration_seq = self.decoder_duration(duration.unsqueeze(2))
        lagtime_seq = self.decoder_lagtime(lagtime.unsqueeze(2)) # lag_time_s
        wnum_seq = self.decoder_wnum(week_num)
        mday_seq = self.decoder_mday(mday)
        hour_seq = self.decoder_hour(hour)
        character_seq = self.decoder_character(character)

        decoder_input = torch.cat([duration_seq, lagtime_seq, character_seq, wnum_seq, mday_seq, hour_seq], dim=-1)
        decoder_input = self.decoder_linear(decoder_input)
        decoder_input = self.decoder_layernorm(decoder_input)
        decoder_input = self.decoder_dropout(decoder_input)

        attn_mask = future_mask(self.max_seq).to(self.device)
        # encoding
        encoding = encoder_input
        for mod in self.encoder:
            encoding = mod(encoding, pos_embed, attn_mask)

        srfe = encoding.clone()
        srfe = self.srfe(srfe, pos_embed, attn_mask)

        # decoding
        decoding = decoder_input
        for mod in self.decoder:
            decoding = mod(decoding, encoding, ltime, srfe, pos_embed,
                           attn_mask, attn_mask)

        predict = self.ffn(decoding, srfe)
        predict = self.layernorm_out(predict + decoding)
        predict = self.classifier(predict)
        preds = self.activation(predict).view(batch_size, -1)
        return preds

        
def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(q, k, v, d_k, positional_bias=None, mask=None, dropout=None,
              memory_decay=False, memory_gamma=None, ltime=None):
    # ltime shape [batch, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [bs, nh, s, s]
    bs, nhead, seqlen = scores.size(0), scores.size(1), scores.size(2)

    if mask is not None:
        mask = mask.unsqueeze(1)

    if memory_decay and memory_gamma is not None and ltime is not None:
        time_seq = torch.cumsum(ltime.float(), dim=-1) - ltime.float()  # [bs, s]
        index_seq = torch.arange(seqlen).unsqueeze(-2).to(q.device)

        dist_seq = time_seq + index_seq

        with torch.no_grad():
            if mask is not None:
                scores_ = scores.masked_fill(mask, 1e-9)
            scores_ = F.softmax(scores_, dim=-1)
            distcum_scores = torch.cumsum(scores_, dim=-1)
            distotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_diff = dist_seq[:, None, :] - dist_seq[:, :, None]
            position_effect = torch.abs(position_diff)[:, None, :, :].type(torch.FloatTensor).to(q.device)
            dist_scores = torch.clamp((distotal_scores - distcum_scores) * position_effect, min=0.)
            dist_scores = dist_scores.sqrt().detach()

        m = nn.Softplus()
        memory_gamma = -1. * m(memory_gamma)
        total_effect = torch.clamp(torch.clamp((dist_scores * memory_gamma).exp(), min=1e-5), max=1e5)
        scores = total_effect * scores

    if positional_bias is not None:
        scores = scores + positional_bias

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    scores = F.softmax(scores, dim=-1)  # [bs, nh, s, s]

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, args, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.h = num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gammas = nn.Parameter(torch.zeros(num_heads, self.args.max_seq_len, 1))
        self.m_srfe = MemorySRFE(embed_dim, num_heads)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, ltime=None, gamma=None, positional_bias=None,
                attn_mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if gamma is not None:
            gamma = self.m_srfe(gamma) + self.gammas
        else:
            gamma = self.gammas

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, positional_bias, attn_mask, self.dropout,
                           memory_decay=True, memory_gamma=gamma, ltime=ltime)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class BaseSRFE(nn.Module):
    def __init__(self, args, in_dim, n_head):
        super(BaseSRFE, self).__init__()
        assert in_dim % n_head == 0
        self.in_dim = in_dim // n_head
        self.n_head = n_head
        self.args = args
        self.attention = MultiHeadAttention(self.args, embed_dim=in_dim, num_heads=n_head, dropout=self.args.drop_out)
        self.dropout = nn.Dropout(self.args.drop_out)
        self.layernorm = nn.LayerNorm(in_dim)

    def forward(self, x, pos_embed, mask):
        out = x
        att_out = self.attention(out, out, out, positional_bias=pos_embed, attn_mask=mask)
        out = out + self.dropout(att_out)
        out = self.layernorm(out)

        return x


class MemorySRFE(nn.Module):
    def __init__(self, in_dim, n_head):
        super(MemorySRFE, self).__init__()
        assert in_dim % n_head == 0
        self.in_dim = in_dim // n_head
        self.n_head = n_head
        self.linear1 = nn.Linear(self.in_dim, 1)

    def forward(self, x):
        bs = x.size(0)

        x = x.view(bs, -1, self.n_head, self.in_dim) \
            .transpose(1, 2) \
            .contiguous()
        x = self.linear1(x)
        return x


class PerformanceSRFE(nn.Module):
    def __init__(self, d_model, d_piv):
        super(PerformanceSRFE, self).__init__()
        self.linear1 = nn.Linear(d_model, 128)
        self.linear2 = nn.Linear(128, d_piv)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)

        return x


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, drop_out):
        super(FFN, self).__init__()
        self.lr1 = nn.Linear(d_model, d_ffn)
        self.act = nn.ReLU()
        self.lr2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.lr1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lr2(x)
        return x


class PivotFFN(nn.Module):
    def __init__(self, d_model, d_ffn, d_piv, drop_out):
        super(PivotFFN, self).__init__()
        self.p_srfe = PerformanceSRFE(d_model, d_piv)
        self.lr1 = nn.Bilinear(d_piv, d_model, d_ffn)
        self.lr2 = nn.Bilinear(d_piv, d_ffn, d_model)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, pivot):
        pivot = self.p_srfe(pivot)

        x = F.gelu(self.lr1(pivot, x))
        x = self.dropout(x)
        x = self.lr2(pivot, x)
        return x


class LANAEncoder(nn.Module):
    def __init__(self, args, d_model, n_heads, d_ffn, max_seq):
        super(LANAEncoder, self).__init__()
        self.max_seq = max_seq
        self.args = args
        self.multi_attn = MultiHeadAttention(self.args, embed_dim=d_model, num_heads=n_heads, dropout=self.args.drop_out)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(self.args.drop_out)
        self.dropout2 = nn.Dropout(self.args.drop_out)

        self.ffn = FFN(d_model, d_ffn, self.args.drop_out)

    def forward(self, x, pos_embed, mask):
        out = x
        att_out = self.multi_attn(out, out, out, positional_bias=pos_embed, attn_mask=mask)
        out = out + self.dropout1(att_out)
        out = self.layernorm1(out)

        ffn_out = self.ffn(out)
        out = self.layernorm2(out + self.dropout2(ffn_out))

        return out


class LANADecoder(nn.Module):
    def __init__(self, args, d_model, n_heads, d_ffn, max_seq):
        super(LANADecoder, self).__init__()
        self.max_seq = max_seq
        self.args = args
        self.multi_attn_1 = MultiHeadAttention(self.args, embed_dim=d_model, num_heads=n_heads, dropout=self.args.drop_out)
        self.multi_attn_2 = MultiHeadAttention(self.args, embed_dim=d_model, num_heads=n_heads, dropout=self.args.drop_out)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(self.args.drop_out)
        self.dropout2 = nn.Dropout(self.args.drop_out)
        self.dropout3 = nn.Dropout(self.args.drop_out)

        self.ffn = FFN(d_model, d_ffn, self.args.drop_out)

    def forward(self, x, memory, ltime, status, pos_embed, mask1, mask2):
        out = x
        att_out_1 = self.multi_attn_1(out, out, out, ltime=ltime,
                                      positional_bias=pos_embed, attn_mask=mask1)
        out = out + self.dropout1(att_out_1)
        out = self.layernorm1(out)

        att_out_2 = self.multi_attn_2(out, memory, memory, ltime=ltime,
                                      gamma=status, positional_bias=pos_embed, attn_mask=mask2)
        out = out + self.dropout2(att_out_2)
        out = self.layernorm2(out)

        ffn_out = self.ffn(out)
        out = self.layernorm3(out + self.dropout3(ffn_out))

        return out


class PositionalBias(nn.Module):
    def __init__(self, args, max_seq, embed_dim, num_heads, bidirectional=True, num_buckets=32):
        super(PositionalBias, self).__init__()
        self.args = args
        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.h = num_heads
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = self.args.max_seq_len

        self.pos_embed = nn.Embedding(max_seq, embed_dim)  # Encoder position Embedding
        self.pos_query_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_key_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_layernorm = nn.LayerNorm(embed_dim)

        self.relative_attention_bias = nn.Embedding(32, self.args.n_heads)

    def forward(self, pos_seq):
        bs = pos_seq.size(0)

        pos_embed = self.pos_embed(pos_seq)
        pos_embed = self.pos_layernorm(pos_embed)

        pos_query = self.pos_query_linear(pos_embed)
        pos_key = self.pos_key_linear(pos_embed)

        pos_query = pos_query.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        pos_key = pos_key.view(bs, -1, self.h, self.d_k).transpose(1, 2)

        absolute_bias = torch.matmul(pos_query, pos_key.transpose(-2, -1)) / math.sqrt(self.d_k)
        relative_position = pos_seq[:, None, :] - pos_seq[:, :, None]

        relative_buckets = 0
        num_buckets = self.num_buckets
        if self.bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_bias = torch.abs(relative_position)
        else:
            relative_bias = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_bias < max_exact

        relative_bias_if_large = max_exact + (
                torch.log(relative_bias.float() / max_exact)
                / math.log(self.max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_bias_if_large = torch.min(
            relative_bias_if_large, torch.full_like(relative_bias_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_bias, relative_bias_if_large)
        relative_position_buckets = relative_buckets.to(pos_seq.device)

        relative_bias = self.relative_attention_bias(relative_position_buckets)
        relative_bias = relative_bias.permute(0, 3, 1, 2).contiguous()

        position_bias = absolute_bias + relative_bias
        return position_bias






def get_model(args, cate_embeddings):
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
    elif args.model == 'lastnquery' : model = LastNQuery(args, cate_embeddings)
    elif args.model == 'lana': model= LANA(args, cate_embeddings)
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
