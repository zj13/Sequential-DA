# -*- coding: utf-8 -*-
# @Time    : 2023/9/19
# @Author  : ZhuangJiabo
# @Email   : zhuangjiaboz@njust.edu.cn

"""
ICLSRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss

import torch.nn.functional as F
def supcon_fake(out1, out2, others, temperature=0.1,):
    N = out1.size(0)

    _out = [out1, out2, others]
    outputs = torch.cat(_out, dim=0)
    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature
    sim_matrix.fill_diagonal_(-5e4)

    mask = torch.zeros_like(sim_matrix)
    mask[2*N:,2*N:] = 1
    mask.fill_diagonal_(0)

    sim_matrix = sim_matrix[2*N:]
    mask = mask[2*N:]
    mask = mask / mask.sum(1, keepdim=True)

    lsm = F.log_softmax(sim_matrix, dim=1)
    lsm = lsm * mask
    d_loss = -lsm.sum(1).mean()
    return d_loss

class COCL(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(COCL, self).__init__(config, dataset)

        self.config = config
        self.nce_fct = nn.CrossEntropyLoss()
        self.start_flag = 0

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.pre_epo = config['pre_epo']
        self.sim_value_first = config['sim_value_first']
        self.sim_value_second = config['sim_value_second']
        self.temperature = config["temperature"]
        self.emb_dropratio = config["emb_dropratio"]
        self.dropout_emb = nn.Dropout(self.emb_dropratio)
        self.dropout_emb2 = nn.Dropout(self.emb_dropratio)
        self.lmd_item = config['lmd_item']
        self.lmd_user = config['lmd_user']
        self.eps = config['eps']
        self.aug_nce_fct = nn.CrossEntropyLoss()
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fn = nn.Linear(self.hidden_size, 1)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len,  seg=0):
        if seg == 0:
            mask = item_seq.gt(0)
            position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            position_embedding = self.position_embedding(position_ids)

            item_emb = self.item_embedding(item_seq)
            input_emb = item_emb + position_embedding
            input_emb = self.LayerNorm(input_emb)  #4096 50 64

            input_emb_drop = self.dropout(input_emb)

            extended_attention_mask = self.get_attention_mask(item_seq)

            trm_output = self.trm_encoder(input_emb_drop, extended_attention_mask, output_all_encoded_layers=True)
            output = trm_output[-1] #4096 50 64
            output = self.gather_indexes(output, item_seq_len - 1)  # [B H] 4096 64
            return output
        '''
        if seg == 1:
            gather_index = (item_seq_len - 1).view(-1, 1, 1).expand(-1, -1, output.shape[-1])
            output_tensor = output.gather(dim=1, index=gather_index)  # 4096 1 64
        # 先计算相似度，再取前几的均值
            output_tensor = torch.transpose(input=output_tensor, dim0=2, dim1=1)  # 4096 64 1
            sim_index = torch.matmul(item_emb, output_tensor).squeeze()  # 4096 50
            sim_index = torch.where(mask, sim_index, -9e15)
            sim_index = torch.softmax(sim_index, dim=1, dtype=torch.float) #importance
            #min_vals, _ = torch.min(sim_index, dim=1, keepdim=True)
            #max_vals, _ = torch.max(sim_index, dim=1, keepdim=True)
            #scaled_x = (sim_index - min_vals) / (max_vals - min_vals)
            mean_value = torch.full_like(sim_index, self.sim_value_first)
            mean_value = mean_value / item_seq_len.unsqueeze(-1)
            tf_sim_index = sim_index >= mean_value  # 4096 50   probility/seq_len
            mask_index = tf_sim_index & mask  # 4096 50
            num_selected = mask_index.sum(dim=-1)  # 4096
            zero_tensor = torch.zeros_like(item_seq)
            aug_seq_sim = torch.where(mask_index, item_seq, zero_tensor) #

            
            result = torch.zeros_like(aug_seq_sim)  #靠前对齐
            # 遍历每个张量
            for i in range(aug_seq_sim.size(0)):
                # 获取当前张量的非零元素索引
                indices = torch.nonzero(aug_seq_sim[i]).squeeze()
                # 获取当前张量的非零元素值
                non_zero_elements = aug_seq_sim[i, indices]
                # 将非零元素放在结果张量的前面
                if non_zero_elements.dim() == 0:
                    non_zero_elements = non_zero_elements.unsqueeze(0)
                result[i, :non_zero_elements.size(0)] = non_zero_elements



            reorder_result = torch.zeros_like(result) #reorder
            indices = torch.nonzero(result)
            values = result[indices[:, 0], indices[:, 1]]
            for i in range(result.size(0)):
                # 获取当前张量的非零元素索引和值
                curr_indices = indices[indices[:, 0] == i]
                curr_values = values[indices[:, 0] == i]

                # 生成随机索引
                random_indices = torch.randperm(curr_values.size(0))

                # 根据随机索引重新排列值
                shuffled_values = curr_values[random_indices]

                # 将重新排列后的值放回到当前张量中
                reorder_result[curr_indices[:, 0], curr_indices[:, 1]] = shuffled_values
            output = self.gather_indexes(output, item_seq_len - 1)

            return output, aug_seq_sim, num_selected
            
            gather_index = (item_seq_len - 1).view(-1, 1, 1).expand(-1, -1, output.shape[-1])
            output_tensor = output.gather(dim=1, index=gather_index)  #4096 1 64
            #先计算相似度，再取前几的均值
            output_tensor = torch.transpose(input=output_tensor, dim0=2, dim1=1) # 4096 64 1
            sim_index = torch.matmul(output,output_tensor)#4096 50 1
            #softmax_sim = F.softmax(sim_index, dim = 1)
            tf_sim_index = sim_index > 60 #4096 50 1
            tf_sim_index = tf_sim_index.squeeze() #4096 50
            mask_index = tf_sim_index & mask #4096 50
            num_selected = mask_index.sum(dim=-1) #4096
            mask_index = mask_index.unsqueeze(2).expand_as(output) #4096 50 64
            zero_tensor = torch.zeros_like(output) #4096 50 64
            sum_tensor = torch.where(mask_index, output, zero_tensor)
            sum_tensor = torch.sum(sum_tensor, 1)
            sum_tensor = sum_tensor / num_selected.unsqueeze(1).expand_as(sum_tensor)# 4096 64
            output = sum_tensor
            #alpha = self.fn(output).to(torch.double) # 4096 50 1
            #alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
            #alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
            #x = self.sess_dropout(item_emb)
            #seq_output = torch.sum(alpha * input_emb, dim=1)
            '''
        #seq_output = F.normalize(seq_output, dim=-1)

    def forward_aug(self, interaction, perturbed = 0):
        item_embedding_all = self.item_embedding.weight
        target_items = interaction[self.POS_ITEM_ID]
        item_embedding_target = item_embedding_all[target_items]
        if perturbed == 1:
            random_noise1 = torch.rand_like(item_embedding_target)
            item_embedding_target = item_embedding_target + torch.sign(item_embedding_target) * F.normalize(random_noise1, dim=-1) * self.eps
        if perturbed == 2:
            random_noise2 = torch.rand_like(item_embedding_target)
            item_embedding_target = item_embedding_target - torch.sign(item_embedding_target) * F.normalize(random_noise2, dim=-1) * self.eps
        return item_embedding_target

    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()

    def ssl_loss(self, anchor_embedding, positive_embedding, negative_embedding=None, all_embedding=None):
        if all_embedding is None:
            all_embedding = torch.cat((positive_embedding, negative_embedding), 0)

        norm_anchor_embedding = F.normalize(anchor_embedding)
        norm_positive_embedding = F.normalize(positive_embedding)
        norm_all_embedding = F.normalize(all_embedding)

        pos_score = torch.mul(norm_anchor_embedding, norm_positive_embedding).sum(dim=1)
        ttl_score = torch.matmul(norm_anchor_embedding, norm_all_embedding.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)

        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def item_CL_global(self, item_seq, item_embs):
        real_item_mask = item_seq != 0  # torch.Size([1024, 50])
        real_item_embs = torch.masked_select(item_embs, real_item_mask.unsqueeze(2)).reshape(-1, item_embs.shape[-1])

        #sample_item_mask = item_sample_seqs !=0
        #sample_item_embs = self.item_embedding(item_sample_seqs)
        #sequence_sum_aug3 = torch.sum(sample_item_embs * sample_item_mask.float().unsqueeze(2),dim=1)  # torch.Size([1024, 64])
        #sequence_mean_aug3 = sequence_sum_aug3 / torch.sum(sample_item_mask, dim=1, keepdim=True)  # torch.Size([1024, 64])
        #sequence_embs_idx = torch.nonzero(real_item_mask)[:, 0]  # torch.Size([X,])
        #sequence_embs = sequence_mean[sequence_embs_idx]  # torch.Size([X, 64])

        #random_noise1 = torch.rand_like(real_item_embs)
        #real_item_embs_aug1 = real_item_embs + torch.sign(real_item_embs) * F.normalize(random_noise1, dim=-1) * self.eps

        random_noise1 = torch.rand_like(item_embs)
        item_embs_aug1 = item_embs + torch.sign(item_embs) * F.normalize(random_noise1, dim=2) * self.eps
        real_item_embs_aug1 = torch.masked_select(item_embs_aug1, real_item_mask.unsqueeze(2)).reshape(-1, item_embs.shape[-1])

        #real_item_embs_aug1 = torch.masked_select(item_embs_aug1, real_item_mask.unsqueeze(2)).reshape(-1, item_embs_aug1.shape[-1])

        #random_noise2 = torch.rand_like(real_item_embs)
        #real_item_embs_aug2 = real_item_embs + torch.sign(real_item_embs) * F.normalize(random_noise2, dim=-1) * self.eps

        random_noise2 = torch.rand_like(item_embs)
        item_embs_aug2 = item_embs + torch.sign(item_embs) * F.normalize(random_noise2, dim=2) * self.eps
        real_item_embs_aug2 = torch.masked_select(item_embs_aug2, real_item_mask.unsqueeze(2)).reshape(-1, item_embs_aug2.shape[-1])


        '''sequence_sum_aug1 = torch.sum(item_embs * real_item_mask.float().unsqueeze(2), dim=1)  # torch.Size([1024, 64])
        sequence_mean_aug1 = sequence_sum_aug1 / torch.sum(real_item_mask, dim=1, keepdim=True)  # torch.Size([1024, 64])
        random_noise4 = torch.rand_like(sequence_mean_aug1)
        sequence_mean_aug1 = sequence_mean_aug1 + + torch.sign(sequence_mean_aug1) * F.normalize(random_noise4, dim=-1) * self.eps


        sequence_sum_aug2 = torch.sum(item_embs * real_item_mask.float().unsqueeze(2), dim=1)  # torch.Size([1024, 64])
        sequence_mean_aug2 = sequence_sum_aug2 / torch.sum(real_item_mask, dim=1, keepdim=True)  # torch.Size([1024, 64])
        random_noise3 = torch.rand_like(sequence_mean_aug2)
        sequence_mean_aug2 = sequence_mean_aug2 + torch.sign(sequence_mean_aug2) * F.normalize(random_noise3, dim=-1) * self.eps'''


        item_CL_global_loss =  self.calculate_cl_loss(real_item_embs_aug1, real_item_embs_aug2) * self.lmd_item
        return item_CL_global_loss

    def user_CL_global(self, item_seq, item_embs, item_sample_seqs):
        real_item_mask = item_seq != 0  # torch.Size([1024, 50])
        #real_item_embs = torch.masked_select(item_embs, real_item_mask.unsqueeze(2)).reshape(-1, item_embs.shape[-1])

        sample_item_mask = item_sample_seqs !=0
        sample_item_embs = self.item_embedding(item_sample_seqs)
        sequence_sum_aug3 = torch.sum(sample_item_embs * sample_item_mask.float().unsqueeze(2),dim=1)  # torch.Size([1024, 64])
        sequence_mean_aug3 = sequence_sum_aug3 / torch.sum(sample_item_mask, dim=1, keepdim=True)  # torch.Size([1024, 64])
        #sequence_embs_idx = torch.nonzero(real_item_mask)[:, 0]  # torch.Size([X,])
        #sequence_embs = sequence_mean[sequence_embs_idx]  # torch.Size([X, 64])

        #random_noise1 = torch.rand_like(real_item_embs)
        #real_item_embs_aug1 = real_item_embs + torch.sign(real_item_embs) * F.normalize(random_noise1, dim=-1) * self.eps

        #random_noise1 = torch.rand_like(item_embs)
        #item_embs_aug1 = item_embs + torch.sign(item_embs) * F.normalize(random_noise1, dim=2) * self.eps
        sequence_sum_aug4 = torch.sum(item_embs * real_item_mask.float().unsqueeze(2), dim=1)  # torch.Size([1024, 64])
        sequence_mean_aug4 = sequence_sum_aug4 / torch.sum(real_item_mask, dim=1, keepdim=True)  # torch.Size([1024, 64])
        #item_embs_aug1 = self.dropout_emb(item_embs)
        #real_item_embs_aug1 = torch.masked_select(item_embs_aug1, real_item_mask.unsqueeze(2)).reshape(-1, item_embs.shape[-1])

        #real_item_embs_aug1 = torch.masked_select(item_embs_aug1, real_item_mask.unsqueeze(2)).reshape(-1, item_embs_aug1.shape[-1])

        #random_noise2 = torch.rand_like(real_item_embs)
        #real_item_embs_aug2 = real_item_embs + torch.sign(real_item_embs) * F.normalize(random_noise2, dim=-1) * self.eps

        #random_noise2 = torch.rand_like(item_embs)
        #item_embs_aug2 = item_embs + torch.sign(item_embs) * F.normalize(random_noise2, dim=2) * self.eps
       # sequence_sum_aug2 = torch.sum(item_embs_aug2 * real_item_mask.float().unsqueeze(2), dim=1)  # torch.Size([1024, 64])
        #sequence_mean_aug2 = sequence_sum_aug2 / torch.sum(real_item_mask, dim=1, keepdim=True)  # torch.Size([1024, 64])
        #item_embs_aug2 = self.dropout_emb2(item_embs)
        #real_item_embs_aug2 = torch.masked_select(item_embs_aug2, real_item_mask.unsqueeze(2)).reshape(-1, item_embs_aug2.shape[-1])
        #sequence_mean = torch.cat((sequence_mean_aug1, sequence_mean_aug2), dim=0)

        user_CL_global_loss =  self.calculate_cl_loss(sequence_mean_aug3, sequence_mean_aug4) * self.lmd_user
        return user_CL_global_loss

    def info_nce(self, z_i, z_j, temp, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # if sim == 'cos':
        #     sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        # elif sim == 'dot':
        sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # if batch_size != self.config['train_batch_size']:
        #     mask = self.mask_correlated_samples(batch_size)
        # else:
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels


    def calculate_tem_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_embs = self.item_embedding(item_seq)
        #seq_output, aug_seq_sim, aug_seq_sim_len = self.forward(item_seq, item_seq_len, 1)
        #seq_output = self.forward(item_seq, item_seq_len)
        #seq_output_1 = self.forward(aug_seq_sim,aug_seq_sim_len)
        #seq_output_2 = self.forward(aug_seq_sim2,item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        #nce_logits, nce_labels = self.info_nce(seq_output_1, seq_output_2, temp=self.temperature,
          #                                     batch_size=item_seq_len.shape[0])

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            #pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            #neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            #loss = self.loss_fct(pos_score, neg_score)
            # return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            #test_item_emb = self.item_dropout(test_item_emb)
            #test_item_emb = F.normalize(test_item_emb, dim=-1)
            #logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            #loss = self.loss_fct(logits, pos_items)
            #nce_loss = self.nce_fct(nce_logits, nce_labels)
            #perturbed_item_embs_1 =self.forward_aug(interaction, 1)
            #perturbed_item_embs_2 = self.forward_aug(interaction, 2)
            sem_aug = interaction['sem_aug']
            user_cl_loss = self.user_CL_global(item_seq, item_embs, sem_aug)
        return  user_cl_loss

    def calculate_loss_CE(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            # return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            item_embs = self.item_embedding(item_seq)
            #sem_aug = interaction['sem_aug']
            #item_cl_loss = self.item_CL_global(item_seq, item_embs)
            #loss += item_cl_loss


            return loss

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B] skip
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))   # [B n_items]
        return scores

    def info_nce(self, z_i, z_j, temp, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # if sim == 'cos':
        #     sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        # elif sim == 'dot':
        sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # if batch_size != self.config['train_batch_size']:
        #     mask = self.mask_correlated_samples(batch_size)
        # else:
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward_captum(self, item_seq, item_seq_emb, item_seq_len):  # item_seq size is (2,50)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # item_emb = self.item_embedding(item_seq)
        item_emb = item_seq_emb
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def update_start(self, start_flag):
        self.start_flag = start_flag