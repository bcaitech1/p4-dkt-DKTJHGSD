import copy
import math

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


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
        relative_bias = relative_bias.permute(0, 3, 1, 2)

        position_bias = absolute_bias + relative_bias
        return position_bias


class LANA(nn.Module):
    def __init__(self, args, cate_embeddings, n_encoder=1, n_decoder=1):
        super(LANA, self).__init__()
        self.args = args
        self.max_seq = self.args.max_seq_len
        self.c_emb = cate_embeddings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.hd_div = args.hd_divider

        self.pos_embed = PositionalBias(self.args, self.max_seq, self.args.hidden_dim, self.args.n_heads, bidirectional=False, num_buckets=32)

        self.encoder_resp_embed = nn.Embedding(3, self.args.hidden_dim//self.hd_div,
                                               padding_idx=0)  # Answer Embedding, 0 for padding
        self.encoder_testid = nn.Embedding(cate_embeddings['testId']+1, self.args.hidden_dim//self.hd_div,
                                              padding_idx=0)  # Exercise ID Embedding, 0 for padding
        self.encoder_assid = nn.Embedding(cate_embeddings['assessmentItemID']+1, self.args.hidden_dim//self.hd_div,
                                              padding_idx=0)  # Exercise ID Embedding, 0 for padding                                      
        self.encoder_part = nn.Embedding(cate_embeddings['part']+1, self.args.hidden_dim//self.hd_div,
                                               padding_idx=0)  # Part Embedding, 0 for padding
        self.encoder_tag = nn.Embedding(cate_embeddings['KnowledgeTag'],self.args.hidden_dim//self.hd_div, padding_idx=0)
        self.encoder_linear = nn.Linear(4 * (self.args.hidden_dim//self.hd_div), self.args.hidden_dim)
        self.encoder_layernorm = nn.LayerNorm(self.args.hidden_dim)
        self.encoder_dropout = nn.Dropout(self.args.drop_out)

        self.decoder_resp_embed = nn.Embedding(3, self.args.hidden_dim//self.hd_div,
                                               padding_idx=0)  # Answer Embedding, 0 for padding
        self.decoder_duration = nn.Sequential(nn.Linear(1, self.args.hidden_dim//self.hd_div),
                                              nn.LayerNorm(self.args.hidden_dim//self.hd_div))
        self.decoder_lagtime = nn.Sequential(nn.Linear(1, self.args.hidden_dim//self.hd_div),
                                              nn.LayerNorm(self.args.hidden_dim//self.hd_div))
        self.decoder_interaction = nn.Embedding(3, self.args.hidden_dim//self.hd_div,
                                                  padding_idx=0)  
        self.decoder_linear = nn.Linear(3 * (self.args.hidden_dim//self.hd_div), self.args.hidden_dim)
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
        correct = input[-1]
        correct = correct.long()
        testid = input[9]
        assid = input[10]
        part = input[12]
        tag = input[11]

        duration = input[0]
        lagtime = input[1]
        interaction = input[-2]
        batch_size = interaction.size(0)

        ltime = lagtime.clone()

        pos_embed = self.pos_embed(self.get_pos_seq().to(self.device))

        # encoder embedding
        #e_correct_seq = self.encoder_resp_embed(correct) # correctness
        testid_seq = self.encoder_testid(testid) # content id
        assid_seq = self.encoder_assid(assid) 
        part_seq = self.encoder_part(part) # part
        tag_seq = self.encoder_tag(tag)

        encoder_input = torch.cat([part_seq, assid_seq, testid_seq, tag_seq], dim=-1)
        encoder_input = self.encoder_linear(encoder_input)
        encoder_input = self.encoder_layernorm(encoder_input)
        encoder_input = self.encoder_dropout(encoder_input)

        # decoder embedding
        #d_correct_seq = self.decoder_resp_embed(correct) # correctness
        duration_seq = self.decoder_duration(duration.unsqueeze(2))
        lagtime_seq = self.decoder_lagtime(lagtime.unsqueeze(2)) # lag_time_s
        interaction_seq = self.decoder_interaction(interaction)
        decoder_input = torch.cat([duration_seq, lagtime_seq, interaction_seq], dim=-1)
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
        #return predict.squeeze(-1)