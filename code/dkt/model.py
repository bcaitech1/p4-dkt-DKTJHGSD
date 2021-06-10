import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import os
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
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.hd_div, padding_idx=0) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_cate = nn.ModuleList([nn.Embedding(cate_embeddings[i]+1, self.hidden_dim//self.hd_div, padding_idx = 0) for i in cate_embeddings])

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
        self.each_cont_idx = []

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
        cate_embeddings['testId'] = 61838
        cate_embeddings['assessmentItemID'] = 4934589

        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.hd_div, padding_idx=0) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_cate = nn.ModuleList([nn.Embedding(cate_embeddings[i]+1, self.hidden_dim//self.hd_div, padding_idx = 0) for i in cate_embeddings])

        # 연속형 Embedding
        self.embedding_cont = nn.ModuleList([nn.Sequential(nn.Linear(i, self.hidden_dim//self.hd_div), 
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

        q = self.query(embed).permute(1, 0, 2)
        
        
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        
        
        
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        out, _ = self.attn(q, k, v)
        
        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
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

        
def get_model(args, cate_embeddings): # junho
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args, cate_embeddings)
    elif args.model == 'lstmattn': model = LSTMATTN(args, cate_embeddings)
    elif args.model == 'bert': model = Bert(args, cate_embeddings)
    elif args.model == 'convbert': model= ConvBert(args, cate_embeddings) # chanhyeong
    elif args.model == 'lastquery': model= LastQuery(args, cate_embeddings) # seoyoon
    return model


def load_model(args, file_name, cate_embeddings):
    model_path = os.path.join(args.model_dir, file_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args, cate_embeddings)

    # 1. load model state
    model.load_state_dict(load_state, strict=True)
   
    print("Loading Model from:", model_path, "...Finished.")
    return model