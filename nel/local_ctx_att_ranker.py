import torch
import torch.nn as nn
import torch.nn.functional as F
import nel.utils as utils
from nel.abstract_word_entity import AbstractWordEntity


class LocalCtxAttRanker(AbstractWordEntity):
    """
    local model with context token attention (from G&H's EMNLP paper)
    """

    def __init__(self, config):
        # nn.Embedding is a simple lookup table that stores embeddings of a fixed dictionary and size.
        #
        # This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of
        # indices, and the output is the corresponding word embeddings.
        config['word_embeddings_class'] = nn.Embedding
        config['entity_embeddings_class'] = nn.Embedding
        super(LocalCtxAttRanker, self).__init__(config)

        self.hid_dims = config['hid_dims']
        # tok_top_n===number of top contextual words for the local model, default=25
        self.tok_top_n = config['tok_top_n']
        # default=0.01
        self.margin = config['margin']

        self.aet_ctx_diag = nn.Parameter(torch.ones(self.aet_dims))
        ## attention diagonal matrix  A in (Ganea and Hofmann, 2017)
        self.att_mat_diag = nn.Parameter(torch.ones(self.emb_dims))
        # ranking contextual words, B in (Ganea and Hofmann, 2017)
        self.tok_score_mat_diag = nn.Parameter(torch.ones(self.emb_dims))
        # local context dropout rate
        self.local_ctx_dr = nn.Dropout(p=0)
        # f function in (Ganea and Hofmann, 2017), input is 2 scalars: psi score and p_e_m
        self.score_combine_linear_1 = nn.Linear(3, self.hid_dims)
        self.score_combine_act_1 = nn.ReLU()
        self.score_combine_linear_2 = nn.Linear(self.hid_dims, 1)

    def print_weight_norm(self):
        print('att_mat_diag', self.att_mat_diag.data.norm())
        print('tok_score_mat_diag', self.tok_score_mat_diag.data.norm())
        print('f - l1.w, b', self.score_combine_linear_1.weight.data.norm(),  self.score_combine_linear_1.bias.data.norm())
        print('f - l2.w, b', self.score_combine_linear_2.weight.data.norm(),  self.score_combine_linear_2.bias.data.norm())

    def print_attention(self, gold_pos):
        token_ids = self._token_ids.data.cpu().numpy()
        entity_ids = self._entity_ids.data.cpu().numpy()
        att_probs = self._att_probs.data.cpu().numpy()
        top_tok_att_ids = self._top_tok_att_ids.data.cpu().numpy()
        gold_pos = gold_pos.data.cpu().numpy()
        scores = self._scores.data.cpu().numpy()

        print('===========================================')
        for tids, eids, ap, aids, gpos, ss in zip(token_ids, entity_ids, att_probs, top_tok_att_ids, gold_pos, scores):
            selected_tids = tids[aids]
            print('-------------------------------')
            print(utils.tokgreen(repr([(self.entity_voca.id2word[e], s) for e, s in zip(eids, ss)])),
                  utils.tokblue(repr(self.entity_voca.id2word[eids[gpos]] if gpos > -1 else 'UNKNOWN')))
            print([(self.word_voca.id2word[t], a[0]) for t, a in zip(selected_tids, ap)])

    def forward(self, token_ids, tok_mask, entity_ids, entity_mask, aet_word_ids, aet_word_mask, p_e_m=None):
        batchsize, n_words = token_ids.size()
        n_entities = entity_ids.size(1)
        tok_mask = tok_mask.view(batchsize, 1, -1)

        # dimensions:  batchsize, n_words, embed_dims
        tok_vecs = self.word_embeddings(token_ids)
        entity_vecs = self.entity_embeddings(entity_ids)
        aet_ent_vec = self.aet_entity_embeddings(entity_ids)
        aet_ctx_vec = self.aet_word_embeddings(aet_word_ids)

        # torch.bmm Performs a batch matrix-matrix product of matrices stored in input and mat2.
        # input and mat2 must be 3-D tensors each containing the same number of matrices.
        # torch.Tensor.permute Permute the dimensions of this tensor.
        # dimension 0: batch, dimension 1:contextual words, dimension2:embedding-dim
        ent_tok_att_scores = torch.bmm(entity_vecs * self.att_mat_diag, tok_vecs.permute(0, 2, 1))
        ent_tok_att_scores = (ent_tok_att_scores * tok_mask).add_((tok_mask - 1).mul_(1e10))
        # max score across several entities, Equation 2 in (Ganea and Hofmann, 2017)
        # dim0: batch,  dim1: entity candidates, dim2: contextual words
        tok_att_scores, _ = torch.max(ent_tok_att_scores, dim=1)

        # select top K, across dimension1: contextual words
        top_tok_att_scores, top_tok_att_ids = torch.topk(tok_att_scores, dim=1, k=min(self.tok_top_n, n_words))
        # torch.functional.softmax Applies a softmax function.
        att_probs = F.softmax(top_tok_att_scores, dim=1).view(batchsize, -1, 1)
        # reshaped,
        att_probs = att_probs / torch.sum(att_probs, dim=1, keepdim=True)

        # embeddings of selected contextual words by id
        selected_tok_vecs = torch.gather(tok_vecs, dim=1,
                                         index=top_tok_att_ids.view(batchsize, -1, 1).repeat(1, 1, tok_vecs.size(2)))
        ctx_vecs = torch.sum((selected_tok_vecs * self.tok_score_mat_diag) * att_probs, dim=1, keepdim=True)
        # apply dropout
        ctx_vecs = self.local_ctx_dr(ctx_vecs)
        ent_ctx_scores = torch.bmm(entity_vecs, ctx_vecs.permute(0, 2, 1)).view(batchsize, n_entities)
        # anonymous entities type words
        aet_ctx_vec = torch.sum(aet_ctx_vec, dim=1, keepdim=True)
        aet_ent_score = torch.bmm(aet_ent_vec * self.aet_ctx_diag, aet_ctx_vec.permute(0, 2, 1)).view(batchsize, n_entities)

        # combine with p(e|m) if p_e_m is not None
        if p_e_m is not None:
            inputs = torch.cat([ent_ctx_scores.view(batchsize * n_entities, -1),
                                aet_ent_score.view(batchsize * n_entities, -1),
                                torch.log(p_e_m + 1e-20).view(batchsize * n_entities, -1)], dim=1)
            hidden = self.score_combine_linear_1(inputs)
            hidden = self.score_combine_act_1(hidden)
            scores = self.score_combine_linear_2(hidden).view(batchsize, n_entities)
        else:
            scores = ent_ctx_scores

        scores = (scores * entity_mask).add_((entity_mask - 1).mul_(1e10))

        # printing attention (debugging)
        self._token_ids = token_ids
        self._entity_ids = entity_ids
        self._att_probs = att_probs
        self._top_tok_att_ids = top_tok_att_ids
        self._scores = scores

        self._entity_vecs = entity_vecs
        self._aet_ent_vec = aet_ent_vec
        self._local_ctx_vecs = ctx_vecs

        return scores

    def regularize(self, max_norm=1):
        l1_w_norm = self.score_combine_linear_1.weight.norm()
        l1_b_norm = self.score_combine_linear_1.bias.norm()
        l2_w_norm = self.score_combine_linear_2.weight.norm()
        l2_b_norm = self.score_combine_linear_2.bias.norm()

        if (l1_w_norm > max_norm).data.all():
            self.score_combine_linear_1.weight.data = self.score_combine_linear_1.weight.data * max_norm / l1_w_norm.data
        if (l1_b_norm > max_norm).data.all():
            self.score_combine_linear_1.bias.data = self.score_combine_linear_1.bias.data *  max_norm / l1_b_norm.data
        if (l2_w_norm > max_norm).data.all():
            self.score_combine_linear_2.weight.data = self.score_combine_linear_2.weight.data * max_norm / l2_w_norm.data
        if (l2_b_norm > max_norm).data.all():
            self.score_combine_linear_2.bias.data = self.score_combine_linear_2.bias.data *  max_norm / l2_b_norm.data

    def loss(self, scores, true_pos):
        loss = F.multi_margin_loss(scores, true_pos, margin=self.margin)
        return loss
