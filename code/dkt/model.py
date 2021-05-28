import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import os

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel    


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 범주형 Embedding 
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        
        self.embedding_character = nn.Embedding(self.args.n_character + 1, self.hidden_dim//3)
        self.embedding_difficulty = nn.Embedding(self.args.n_difficulty + 1, self.hidden_dim//3)


        # 수치형 Embedding
        self.embedding_duration = nn.Sequential(nn.Linear(1, self.hidden_dim//3), # 연속형 feature = duration 1개 so in_features == 1
                                                nn.LayerNorm(self.hidden_dim//3))

        # # embedding combination projection

        self.comb_proj = nn.Linear((self.hidden_dim//3)*7, self.hidden_dim)

        self.embedding_duration = nn.Embedding(self.args.n_duration + 1, self.hidden_dim//3)
        #self.embedding_difficulty = nn.Embedding(self.args.n_difficulty + 1, self.hidden_dim//3)

        # # embedding combination projection
        #self.comb_proj = nn.Linear((self.hidden_dim//3)*6, self.hidden_dim)
        self.comb_proj = nn.Linear((self.hidden_dim//3)*5, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

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
        duration, test, question, tag, _, mask, interaction, character, difficulty = input

        batch_size = interaction.size(0)

        # 범주형 Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        
        embed_character = self.embedding_character(character)
        embed_difficulty = self.embedding_difficulty(difficulty)

        # 수치형 embedding
        embed_duration = self.embedding_duration(duration.unsqueeze(2)) # duration to [batch, seq, # cont features]
      

#         embed = torch.cat([embed_interaction,
#                            embed_test,
#                            embed_question,
#                            embed_tag,
#                            embed_duration,
#                            embed_difficulty], 2)

#         embed = torch.cat([embed_interaction,
#                            embed_test,
#                            embed_question,
#                            embed_tag,
#                            embed_duration,
#                            embed_difficulty], 2)

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,
                           embed_character,
                           embed_difficulty,
                           embed_duration],2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):

    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_character = nn.Embedding(self.args.n_character + 1, self.hidden_dim // 3)
        self.embedding_difficulty = nn.Embedding(self.args.n_difficulty + 1, self.hidden_dim//3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 6, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

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
        duration, test, question, tag, _, mask, interaction, character, difficulty = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_character = self.embedding_character(character)
        embed_difficulty = self.embedding_difficulty(difficulty)

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,
                           embed_character,
                           embed_difficulty ], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds

class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

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
        test, question, tag, _, mask, interaction, duration = input
        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_duration = self.embedding_duration(duration)

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,
                           embed_duration], 2)

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds

class ConvBert(nn.Module): # chanhyeong

    def __init__(self, args):
        super(ConvBert, self).__init__()
        self.args = args
        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        self.embedding_duration = nn.Embedding(self.args.n_duration + 1, self.hidden_dim//3)
        
        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*5, self.hidden_dim)

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
        test, question, tag, _, mask, interaction, duration,_ = input
        batch_size = interaction.size(0)

        # 신나는 embedding
        
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_duration = self.embedding_duration(duration)
        
        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,
                           embed_duration], 2)

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

class ConvBert(nn.Module): # chanhyeong
    def __init__(self, args):
        super(ConvBert, self).__init__()
        self.args = args
        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # 범주형 Embedding 
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        self.embedding_character = nn.Embedding(self.args.n_character + 1, self.hidden_dim//3)
        self.embedding_difficulty = nn.Embedding(self.args.n_difficulty + 1, self.hidden_dim//3)

        # 수치형 Embedding
        self.embedding_duration = nn.Sequential(nn.Linear(1, self.hidden_dim//3), # 연속형 feature = duration 1개 so in_features == 1
                                                nn.LayerNorm(self.hidden_dim//3))

        # # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*7, self.hidden_dim)


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
        duration, test, question, tag, _, mask, interaction, character, difficulty = input
        batch_size = interaction.size(0)

        # 범주형 Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_character = self.embedding_character(character)
        embed_difficulty = self.embedding_difficulty(difficulty)

        # 수치형 embedding
        embed_duration = self.embedding_duration(duration.unsqueeze(2)) # duration to [batch, seq, # cont features]

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,
                           embed_character,
                           embed_difficulty,
                           embed_duration], 2)

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


def get_model(args): # junho
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args)
    elif args.model == 'lstmattn': model = LSTMATTN(args)
    elif args.model == 'bert': model = Bert(args)
    elif args.model == 'convbert': model= ConvBert(args) # chanhyeong
    return model

    

def load_model(args):
    model_path = os.path.join(args.model_dir, f'{args.model}.pt')
    print("Loading Model from:", model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(torch.load(model_path))
   
    print("Loading Model from:", model_path, "...Finished.")
    return model