import torch.nn as nn
import torch.nn.functional as F

from RAdam import *

from config import Config
import numpy as np
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from deepctr_torch.layers.interaction import InteractingLayer



class PredictLayer(nn.Module):
    def __init__(self, n_categories, n_fc, h_dim, dropout=0):
        super(PredictLayer, self).__init__()
        self.n_fc = n_fc
        self.h_dim = h_dim

        self.fc_a = nn.Linear(n_fc//Config.CON, 1)

        self.fc_theta = nn.Linear(n_fc+n_fc//Config.CON, self.h_dim)
        self.norm_layer = nn.LayerNorm(n_fc//Config.CON)
        self.fc_theta2 = nn.Linear(self.h_dim, self.h_dim//2)
        self.fc_theta3 = nn.Linear(self.h_dim//2, 1)

        self.theta_b = nn.Parameter(torch.rand(1))
        self.dropout = nn.Dropout(dropout)
        self.ls_w = nn.Parameter(torch.rand(1))
        self.lm_w = nn.Parameter(torch.rand(1))
        self.ld_w = nn.Parameter(torch.rand(1))

        self.l1 = nn.Parameter(torch.rand(1))

    def forward(self, x, cat_params, categories, lt_s, lt_m, lt_d):

        b_p = cat_params['b_p']  # bs sl 1
        c_emb = cat_params['c_weighted']

        c_emb_input = self.norm_layer(c_emb)
        a_p = F.relu(self.fc_a(c_emb_input))
        a_p = 4 * torch.sigmoid(a_p)  # 0 - 4

        # x = self.norm_layers(x)
        in_x = torch.cat([x, c_emb], dim=-1)
        theta = F.relu(self.fc_theta(in_x))
        theta = self.dropout(theta)
        theta = self.fc_theta2(theta)
        theta = self.dropout(theta)
        theta = self.fc_theta3(theta)

        # theta = self.fc_theta(x)
        forget = self.ls_w * torch.exp(-torch.abs(lt_s)) + \
                 self.lm_w * torch.exp(-torch.abs(lt_m)) + \
                 self.ld_w * torch.exp(-torch.abs(lt_d))
        forget = F.softmax(forget,dim=1)
        if Config.Forget:
            theta = (1-self.l1)*theta + self.l1*forget + self.theta_b
        else:
            theta = theta + self.theta_b
        exp = 1.7 * a_p * (theta - b_p)


        output = exp

        return torch.sigmoid(output)


# 两层前馈网络
class FFN(nn.Module):
    def __init__(self, in_feat, ffn_dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(in_feat, in_feat // 2)
        self.linear2 = nn.Linear(in_feat // 2, in_feat)
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        return out

class SENETLayer_weight(nn.Module):
    def __init__(self, filed_size, reduction_ratio=3, seed=1024, device=Config.device):
        super(SENETLayer_weight, self).__init__()
        self.seed = seed
        self.filed_size = filed_size
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(self.filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.filed_size, bias=False),
            nn.ReLU()
        )
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        Z = torch.mean(inputs, dim=-1, out=None)
        A = self.excitation(Z)
        # V = torch.mul(inputs, torch.unsqueeze(A, dim=2))
        return A


class EncoderEmbedding(nn.Module):
    def __init__(self, n_exercises, n_categories, n_responses, n_dims, seq_len, max_cnum, bs, att_layer_num=1,mcakt=False,autoint=True):
        super(EncoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.max_cnum = max_cnum
        self.bsize = bs
        # 不用eid作索引 只用category
        self.category_embed = nn.Embedding(n_categories, n_dims, padding_idx=0)
        self.response_embed = nn.Embedding(n_responses, n_dims)
        self.position_embed = nn.Embedding(seq_len, 2 * n_dims)

        self.b_mlp = nn.Sequential(nn.Linear(n_dims, n_dims),
                                    nn.Linear(n_dims, 1), nn.Tanh())

        self.emb_layer = nn.Linear(2 * n_dims, n_dims)
        self.b_param = nn.Linear(1, n_dims)
        self.b_param_2 = nn.Linear(n_dims, 1)

        self.cl_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")


        if mcakt:
            # AutoInt
            if autoint:
                self.c_AI = nn.ModuleList(
                    [InteractingLayer(self.n_dims, 2, True)
                     for _ in range(att_layer_num)])
            # SENet
            else:
                self.SE = SENETLayer_weight(filed_size=Config.MAX_CATS_PER, reduction_ratio=2, seed=1024,
                                        device=Config.device)


    def forward(self, exercises, categories, cate_num, exe_diff, lt_s, lt_m, lt_d, responses, at_dict=None,mcakt=False,autoint=True):
        # excises: batch_size * 100
        # categories: batch_size * 100
        # e = self.exercise_embed(exercises.long())

        if at_dict is not None:
            perturbation = at_dict['perturbation']

        self.bsize = categories.shape[0]

        r = self.response_embed(responses.long())
        no_res = torch.ones_like(responses).to(Config.device) * 2
        no_r = self.response_embed(no_res.long())
        seq = torch.arange(self.seq_len, device=Config.device).unsqueeze(0)
        p = self.position_embed(seq)

        # 多concept嵌入
        if mcakt:
            # tag = torch.full(categories[:, :, 0].shape, Config.TOTAL_CAT - 1, device=Config.device)
            # categories[:, :, 0] = torch.where(categories[:, :, 0] == 0,
            #                                   tag,
            #                                   categories[:, :, 0])
            f_c = [self.category_embed(categories[:, :, 0])]
            for i in range(1, Config.MAX_CATS_PER):
                f_c.append(self.category_embed(categories[:, :, i]))
            c_a = torch.stack(f_c, dim=-2)  # B_S * SEQ_LEN * 10 * DIM
            t_c = torch.split(c_a, 1, dim=1)  # seq_len * (b_s * 10 * dim)
            t_c = torch.cat(t_c, dim=0).squeeze(1)  # seq_len*b_s  * 10  * dim
            if autoint:
                # attentive concept
                for layer in self.c_AI:
                    t_c = layer(t_c.float())
                t_c = torch.sum(t_c, dim=-2)  # seq_len*b_s  * dim
                t_c = torch.split(t_c, self.bsize, dim=0)  # seq_len * (b_s * dim)
                c_weighted = torch.stack(t_c, dim=1)  # B_S * SEQ_LEN * dim
            else:
                # SENet concept
                t_c_w = self.SE(t_c.float())  # seq_len*b_s  * 10
                t_c_w = torch.split(t_c_w, self.bsize, dim=0)  # seq_len * (b_s * 10)
                c_w = torch.stack(t_c_w, dim=1)  # B_S * SEQ_LEN * 10
                # c_w : B_S * SEQ_LEN * 10
                # c   : B_S * SEQ_LEN * 10 * DIM
                t_c = torch.mul(c_a, torch.unsqueeze(c_w, dim=-1))
                # Residual
                t_c = t_c + c_a
                c_weighted = torch.sum(t_c, dim=-2)
        else:
            # mean concept
            c = self.category_embed(categories[:, :, 0].long())
            for i in range(1, self.max_cnum):
                temp_c = self.category_embed(categories[:, :, i].long())
                c += temp_c
            c_n = cate_num.unsqueeze(-1)
            cn = torch.where(c_n == 0, 1, c_n)
            c_weighted = c/cn
        if at_dict is not None:
            if perturbation is not None:
                c_weighted += perturbation

        b_p = torch.ones(exe_diff.shape).to(Config.device) - exe_diff
        b_p = b_p.to(torch.float32)
        exe_params1 = self.b_param(b_p.unsqueeze(-1))
        exe_params = self.b_param_2(exe_params1)

        if Config.EXEB:
            e_emb = c_weighted + exe_params
        else:
            e_emb = c_weighted


        concept_answer = torch.cat((e_emb, r), 2)
        answer_concept = torch.cat((r, e_emb), 2)
        answer = responses.unsqueeze(2).expand_as(concept_answer)
        concept_answer_embedding = torch.where(answer == 1, concept_answer, answer_concept)
        neg_concept_answer_embedding = torch.where(answer == 1, answer_concept, concept_answer)

        answer_concept_nor = torch.cat((no_r, e_emb), 2)

        if not Config.RoPE:
            return concept_answer_embedding + p, answer_concept_nor, {"b_p": exe_params, "c_weighted": c_weighted,
                                                                  "concept_answer_embedding": concept_answer_embedding,
                                                                  'neg_concept_answer_embedding': neg_concept_answer_embedding + p}

        return concept_answer_embedding, answer_concept_nor, {"b_p": exe_params, "c_weighted": c_weighted, "concept_answer_embedding":concept_answer_embedding, 'neg_concept_answer_embedding':neg_concept_answer_embedding + p}


class PositionalBias(nn.Module):
    def __init__(self, max_seq, embed_dim, num_heads, bidirectional=True, num_buckets=32, max_distance=Config.MAX_SEQ):
        super(PositionalBias, self).__init__()
        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.h = num_heads
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.pos_embed = nn.Embedding(max_seq, embed_dim)  # Encoder position Embedding
        self.pos_query_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_key_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_layernorm = nn.LayerNorm(embed_dim)

        self.relative_attention_bias = nn.Embedding(32, num_heads)

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


def attention(q, k, v, d_k, pad_zero, positional_bias=None, mask=None, dropout=None,
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
        x1 = torch.arange(seqlen).expand(seqlen, -1).to(Config.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            if mask is not None:
                scores_ = scores.masked_fill(mask, -1e9)
            scores_ = F.softmax(scores_, dim=-1)  # bs h sl sl
            scores_ = torch.cat([pad_zero, scores_[:, :, 1:, :]], dim=-2)
            distcum_scores = torch.cumsum(scores_, dim=-1)
            distotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = torch.abs(
                x1-x2)[None, None, :, :].type(torch.FloatTensor).to(Config.device)  # 1, 1, seqlen, seqlen
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
    # 对sl=1 scores置0
    scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=-2)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, kq_same=Config.KQ_SAME, bias=True, rope_v=True):
        super(MultiHeadAttention, self).__init__()

        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.h = num_heads
        self.kq_same = kq_same
        self.proj_bias = bias

        if kq_same is False:
            self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gammas = nn.Parameter(torch.zeros(num_heads, Config.MAX_SEQ, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.rope_v = rope_v
        self.rope = RotaryEmbedding(dim=embed_dim//num_heads) if self.rope_v else None
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.k_linear.weight)
        torch.nn.init.xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            torch.nn.init.xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            torch.nn.init.constant_(self.k_linear.bias, 0.)
            torch.nn.init.constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                torch.nn.init.constant_(self.q_linear.bias, 0.)
            torch.nn.init.constant_(self.out.bias, 0.)

    def forward(self, q, k, v, ltime=None, gamma=None, positional_bias=None,
                attn_mask=None):
        # cache with a key that is the sequence length, so that it does not need to recompute
        if self.rope_v:
            freqs = self.rope(torch.arange(Config.MAX_SEQ).to(Config.device), cache_key=Config.MAX_SEQ)
            freqs = freqs[:Config.MAX_SEQ]
            freqs = freqs[None,None,...]

        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rope_v:
            # apply the rotations to your queries and keys after the heads have been split out
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
            # but prior to the dot product and subsequent softmax (attention)

        gamma = self.gammas
        pad_zero = torch.zeros(bs, self.h, 1, Config.MAX_SEQ).to(Config.device)  # [bs, nh, 1, s]

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, pad_zero, positional_bias, attn_mask, self.dropout,
                           memory_decay=True, memory_gamma=gamma, ltime=ltime)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=0).astype('bool')  # 对角线mask
    return torch.from_numpy(future_mask)


# 多头注意力
class StackedNMultiHeadAttention(nn.Module):
    def __init__(self, n_stacks, n_dims, n_heads, seq_len, n_multihead=1, ma_dropout=0, ffn_dropout=0):
        super(StackedNMultiHeadAttention, self).__init__()
        self.n_stacks = n_stacks
        self.n_multihead = n_multihead
        self.n_dims = n_dims
        self.norm_layers = nn.LayerNorm(n_dims)
        self.max_seq = seq_len
        # n_stacks has n_multiheads each
        self.multihead_layers = nn.ModuleList(
            n_stacks * [nn.ModuleList(n_multihead * [MultiHeadAttention(embed_dim=n_dims,
                                                                        num_heads=n_heads,
                                                                        dropout=ma_dropout,
                                                                        rope_v=Config.RoPE), ]), ])
        self.ffn = nn.ModuleList(n_stacks * [FFN(n_dims,ffn_dropout)])

    def forward(self, input_q, input_k, input_v, pos_embed=None, ltime=None, encoder_output=None, break_layer=None):
        for stack in range(self.n_stacks):

            for multihead in range(self.n_multihead):
                norm_q = self.norm_layers(input_q)
                norm_k = self.norm_layers(input_k)
                norm_v = self.norm_layers(input_v)
                attn_mask = future_mask(self.max_seq).to(Config.device)
                heads_output = self.multihead_layers[stack][multihead](q=norm_q,  # 变换tensor维度
                                                                       k=norm_k,
                                                                       v=norm_v,
                                                                       positional_bias=pos_embed,
                                                                       ltime=ltime,
                                                                       attn_mask=attn_mask)
                # heads_output = heads_output.permute(1, 0, 2)
                # assert encoder_output != None and break_layer is not None
                if encoder_output != None and multihead == break_layer:
                    # 有来自encoder的信息
                    assert break_layer <= multihead, " break layer should be less than multihead layers and postive integer"

                    input_k = input_v = encoder_output
                    input_q = input_q + heads_output
                else:

                    input_q = input_q + heads_output
                    input_k = input_k + heads_output
                    input_v = input_v + heads_output
            last_norm = self.norm_layers(heads_output)
            ffn_output = self.ffn[stack](last_norm)
            ffn_output = ffn_output + heads_output
        # after loops = input_q = input_k = input_v
        return ffn_output
