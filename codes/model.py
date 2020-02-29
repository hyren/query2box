#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import *
import random
import pickle
import math
def Identity(x):
    return x

class SetIntersection(nn.Module):
    def __init__(self, mode_dims, expand_dims, agg_func=torch.min):
        super(SetIntersection, self).__init__()
        self.agg_func = agg_func
        self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.pre_mats)
        self.register_parameter("premat", self.pre_mats)
        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat", self.post_mats)
        self.pre_mats_im = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.pre_mats_im)
        self.register_parameter("premat_im", self.pre_mats_im)
        self.post_mats_im = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats_im)
        self.register_parameter("postmat_im", self.post_mats_im)

    def forward(self, embeds1, embeds2, embeds3 = [], name='real'):
        if name == 'real':
            temp1 = F.relu(embeds1.mm(self.pre_mats))
            temp2 = F.relu(embeds2.mm(self.pre_mats))
            if len(embeds3) > 0:
                temp3 = F.relu(embeds3.mm(self.pre_mats))
                combined = torch.stack([temp1, temp2, temp3])
            else:
                combined = torch.stack([temp1, temp2])
            combined = self.agg_func(combined, dim=0)
            if type(combined) == tuple:
                combined = combined[0]
            combined = combined.mm(self.post_mats)

        elif name == 'img':
            temp1 = F.relu(embeds1.mm(self.pre_mats_im))
            temp2 = F.relu(embeds2.mm(self.pre_mats_im))
            if len(embeds3) > 0:
                temp3 = F.relu(embeds3.mm(self.pre_mats_im))
                combined = torch.stack([temp1, temp2, temp3])
            else:
                combined = torch.stack([temp1, temp2])
            combined = self.agg_func(combined, dim=0)
            if type(combined) == tuple:
                combined = combined[0]
            combined = combined.mm(self.post_mats_im)
        return combined

class CenterSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, agg_func=torch.min, bn='no', nat=1, name='Real_center'):
        super(CenterSet, self).__init__()
        assert nat == 1, 'vanilla method only support 1 nat now'
        self.center_use_offset = center_use_offset
        self.agg_func = agg_func
        self.bn = bn
        self.nat = nat
        if center_use_offset:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims*2, mode_dims))
        else:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))

        nn.init.xavier_uniform(self.pre_mats)
        self.register_parameter("premat_%s"%name, self.pre_mats)
        if bn != 'no':
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn2 = nn.BatchNorm1d(mode_dims)
            self.bn3 = nn.BatchNorm1d(mode_dims)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat_%s"%name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        if self.center_use_offset:
            temp1 = torch.cat([embeds1, embeds1_o], dim=1)
            temp2 = torch.cat([embeds2, embeds2_o], dim=1)
            if len(embeds3) > 0:
                temp3 = torch.cat([embeds3, embeds3_o], dim=1)
        else:
            temp1 = embeds1
            temp2 = embeds2
            if len(embeds3) > 0:
                temp3 = embeds3

        if self.bn == 'no':
            temp1 = F.relu(temp1.mm(self.pre_mats))
            temp2 = F.relu(temp2.mm(self.pre_mats))
        elif self.bn == 'before':
            temp1 = F.relu(self.bn1(temp1.mm(self.pre_mats)))
            temp2 = F.relu(self.bn2(temp2.mm(self.pre_mats)))
        elif self.bn == 'after':
            temp1 = self.bn1(F.relu(temp1.mm(self.pre_mats)))
            temp2 = self.bn2(F.relu(temp2.mm(self.pre_mats)))
        if len(embeds3) > 0:
            if self.bn == 'no':
                temp3 = F.relu(temp3.mm(self.pre_mats))
            elif self.bn == 'before':
                temp3 = F.relu(self.bn3(temp3.mm(self.pre_mats)))
            elif self.bn == 'after':
                temp3 = self.bn3(F.relu(temp3.mm(self.pre_mats)))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats)
        return combined

class MeanSet(nn.Module):
    def __init__(self):
        super(MeanSet, self).__init__()

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        if len(embeds3) > 0:
            return torch.mean(torch.stack([embeds1, embeds2, embeds3], dim=0), dim=0)
        else:
            return torch.mean(torch.stack([embeds1, embeds2], dim=0), dim=0)

class MinSet(nn.Module):
    def __init__(self):
        super(MinSet, self).__init__()

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        if len(embeds3) > 0:
            return torch.min(torch.stack([embeds1, embeds2, embeds3], dim=0), dim=0)[0]
        else:
            return torch.min(torch.stack([embeds1, embeds2], dim=0), dim=0)[0]

class OffsetSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, agg_func=torch.min, name='Real_offset'):
        super(OffsetSet, self).__init__()
        self.offset_use_center = offset_use_center
        self.agg_func = agg_func
        if offset_use_center:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims*2, mode_dims))
            nn.init.xavier_uniform(self.pre_mats)
            self.register_parameter("premat_%s"%name, self.pre_mats)
        else:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
            nn.init.xavier_uniform(self.pre_mats)
            self.register_parameter("premat_%s"%name, self.pre_mats)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform(self.post_mats)
        self.register_parameter("postmat_%s"%name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        if self.offset_use_center:
            temp1 = torch.cat([embeds1, embeds1_o], dim=1)
            temp2 = torch.cat([embeds2, embeds2_o], dim=1)
            if len(embeds3_o) > 0:
                temp3 = torch.cat([embeds3, embeds3_o], dim=1)
        else:
            temp1 = embeds1_o
            temp2 = embeds2_o
            if len(embeds3_o) > 0:
                temp3 = embeds3_o
        temp1 = F.relu(temp1.mm(self.pre_mats))
        temp2 = F.relu(temp2.mm(self.pre_mats))
        if len(embeds3_o) > 0:
            temp3 = F.relu(temp3.mm(self.pre_mats))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats)
        return combined

class InductiveOffsetSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, off_reg, agg_func=torch.min, name='Real_offset'):
        super(InductiveOffsetSet, self).__init__()
        self.offset_use_center = offset_use_center
        self.agg_func = agg_func
        self.off_reg = off_reg
        self.OffsetSet_Module = OffsetSet(mode_dims, expand_dims, offset_use_center, self.agg_func)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        if len(embeds3_o) > 0:
            offset_min = torch.min(torch.stack([embeds1_o, embeds2_o, embeds3_o]), dim=0)[0]
        else:
            offset_min = torch.min(torch.stack([embeds1_o, embeds2_o]), dim=0)[0]
        offset = offset_min * F.sigmoid(self.OffsetSet_Module(embeds1, embeds1_o, embeds2, embeds2_o, embeds3, embeds3_o))
        return offset

class AttentionSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, att_reg=0., att_tem=1., att_type="whole", bn='no', nat=1, name="Real"):
        super(AttentionSet, self).__init__()
        self.center_use_offset = center_use_offset
        self.att_reg = att_reg
        self.att_type = att_type
        self.att_tem = att_tem
        self.Attention_module = Attention(mode_dims, expand_dims, center_use_offset, att_type=att_type, bn=bn, nat=nat)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        temp1 = (self.Attention_module(embeds1, embeds1_o) + self.att_reg)/(self.att_tem+1e-4)
        temp2 = (self.Attention_module(embeds2, embeds2_o) + self.att_reg)/(self.att_tem+1e-4)
        if len(embeds3) > 0:
            temp3 = (self.Attention_module(embeds3, embeds3_o) + self.att_reg)/(self.att_tem+1e-4)
            if self.att_type == 'whole':
                combined = F.softmax(torch.cat([temp1, temp2, temp3], dim=1), dim=1)
                center = embeds1*(combined[:,0].view(embeds1.size(0), 1)) + \
                        embeds2*(combined[:,1].view(embeds2.size(0), 1)) + \
                        embeds3*(combined[:,2].view(embeds3.size(0), 1))
            elif self.att_type == 'ele':
                combined = F.softmax(torch.stack([temp1, temp2, temp3]), dim=0)
                center = embeds1*combined[0] + embeds2*combined[1] + embeds3*combined[2]
        else:
            if self.att_type == 'whole':
                combined = F.softmax(torch.cat([temp1, temp2], dim=1), dim=1)
                center = embeds1*(combined[:,0].view(embeds1.size(0), 1)) + \
                        embeds2*(combined[:,1].view(embeds2.size(0), 1))
            elif self.att_type == 'ele':
                combined = F.softmax(torch.stack([temp1, temp2]), dim=0)
                center = embeds1*combined[0] + embeds2*combined[1]

        return center

class Attention(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, att_type, bn, nat, name="Real"):
        super(Attention, self).__init__()
        self.center_use_offset = center_use_offset
        self.bn = bn
        self.nat = nat
        if center_use_offset:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims*2, mode_dims))
        else:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform(self.atten_mats1)
        self.register_parameter("atten_mats1_%s"%name, self.atten_mats1)
        if self.nat >= 2:
            self.atten_mats1_1 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform(self.atten_mats1_1)
            self.register_parameter("atten_mats1_1_%s"%name, self.atten_mats1_1)
        if self.nat >= 3:
            self.atten_mats1_2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform(self.atten_mats1_2)
            self.register_parameter("atten_mats1_2_%s"%name, self.atten_mats1_2)
        if bn != 'no':
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn1_1 = nn.BatchNorm1d(mode_dims)
            self.bn1_2 = nn.BatchNorm1d(mode_dims)
        if att_type == 'whole':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, 1))
        elif att_type == 'ele':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
        nn.init.xavier_uniform(self.atten_mats2)
        self.register_parameter("atten_mats2_%s"%name, self.atten_mats2)

    def forward(self, center_embed, offset_embed=None):
        if self.center_use_offset:
            temp1 = torch.cat([center_embed, offset_embed], dim=1)
        else:
            temp1 = center_embed
        if self.nat >= 1:
            if self.bn == 'no':
                temp2 = F.relu(temp1.mm(self.atten_mats1))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1(temp1.mm(self.atten_mats1)))
            elif self.bn == 'after':
                temp2 = self.bn1(F.relu(temp1.mm(self.atten_mats1)))
        if self.nat >= 2:
            if self.bn == 'no':
                temp2 = F.relu(temp2.mm(self.atten_mats1_1))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1_1(temp2.mm(self.atten_mats1_1)))
            elif self.bn == 'after':
                temp2 = self.bn1_1(F.relu(temp2.mm(self.atten_mats1_1)))
        if self.nat >= 3:
            if self.bn == 'no':
                temp2 = F.relu(temp2.mm(self.atten_mats1_2))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1_2(temp2.mm(self.atten_mats1_2)))
            elif self.bn == 'after':
                temp2 = self.bn1_2(F.relu(temp2.mm(self.atten_mats1_2)))
        temp3 = temp2.mm(self.atten_mats2)
        return temp3

class Query2box(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 writer=None, geo=None, 
                 cen=None, offset_deepsets=None,
                 center_deepsets=None, offset_use_center=None, center_use_offset=None,
                 att_reg = 0., off_reg = 0., att_tem = 1., euo = False, 
                 gamma2=0, bn='no', nat=1, activation='relu'):
        super(Query2box, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.writer=writer
        self.geo = geo
        self.cen = cen
        self.offset_deepsets = offset_deepsets
        self.center_deepsets = center_deepsets
        self.offset_use_center = offset_use_center
        self.center_use_offset = center_use_offset
        self.att_reg = att_reg
        self.off_reg = off_reg
        self.att_tem = att_tem
        self.euo = euo
        self.his_step = 0
        self.bn = bn
        self.nat = nat
        if activation == 'none':
            self.func = Identity
        elif activation == 'relu':
            self.func = F.relu
        elif activation == 'softplus':
            self.func = F.softplus
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        if gamma2 == 0:
            gamma2 = gamma

        self.gamma2 = nn.Parameter(
            torch.Tensor([gamma2]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        if self.geo == 'vec':
            if self.center_deepsets == 'vanilla':
                self.deepsets = CenterSet(self.relation_dim, self.relation_dim, False, agg_func = torch.mean, bn=bn, nat=nat)
            elif self.center_deepsets == 'attention':
                self.deepsets = AttentionSet(self.relation_dim, self.relation_dim, False, 
                                                    att_reg = self.att_reg, att_tem = self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'eleattention':
                self.deepsets = AttentionSet(self.relation_dim, self.relation_dim, False, 
                                                    att_reg = self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'mean':
                self.deepsets = MeanSet()
            else:
                assert False

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding, 
                a=0., 
                b=self.embedding_range.item()
            )
            if self.euo:
                self.entity_offset_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
                nn.init.uniform_(
                    tensor=self.entity_offset_embedding, 
                    a=0., 
                    b=self.embedding_range.item()
                )

            if self.center_deepsets == 'vanilla':
                self.center_sets = CenterSet(self.relation_dim, self.relation_dim, self.center_use_offset, agg_func = torch.mean, bn=bn, nat=nat)
            elif self.center_deepsets == 'attention':
                self.center_sets = AttentionSet(self.relation_dim, self.relation_dim, self.center_use_offset, 
                                                    att_reg = self.att_reg, att_tem = self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'eleattention':
                self.center_sets = AttentionSet(self.relation_dim, self.relation_dim, self.center_use_offset, 
                                                    att_reg = self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'mean':
                self.center_sets = MeanSet()
            else:
                assert False

            if self.offset_deepsets == 'vanilla':
                self.offset_sets = OffsetSet(self.relation_dim, self.relation_dim, self.offset_use_center, agg_func = torch.mean)
            elif self.offset_deepsets == 'inductive':
                self.offset_sets = InductiveOffsetSet(self.relation_dim, self.relation_dim, self.offset_use_center, self.off_reg, agg_func=torch.mean)
            elif self.offset_deepsets == 'min':
                self.offset_sets = MinSet()
            else:
                assert False

        if model_name not in ['TransE', 'BoxTransE']:
            raise ValueError('model %s not supported' % model_name)
            
    def forward(self, sample, rel_len, qtype, mode='single'):
        if qtype == 'chain-inter':
            assert mode == 'tail-batch'
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
            head = torch.cat([head_1, head_2], dim=0)
            if self.euo and self.geo == 'box':
                head_offset_1 = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
                head_offset = torch.cat([head_offset_1, head_offset_2], dim=0)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

            relation_11 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1)
            relation_12 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1).unsqueeze(1)
            relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1).unsqueeze(1)
            relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

            if self.geo == 'box':
                offset_11 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1)
                offset_12 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1).unsqueeze(1)
                offset_2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1).unsqueeze(1)
                offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

        elif qtype == 'inter-chain' or qtype == 'union-chain':
            assert mode == 'tail-batch'
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
            head = torch.cat([head_1, head_2], dim=0)
            if self.euo and self.geo == 'box':
                head_offset_1 = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                head_offset = torch.cat([head_offset_1, head_offset_2], dim=0)

            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

            relation_11 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1)
            relation_12 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1).unsqueeze(1)
            relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1).unsqueeze(1)
            relation = torch.cat([relation_11, relation_12, relation_2], dim=0)

            if self.geo == 'box':
                offset_11 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1)
                offset_12 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1).unsqueeze(1)
                offset_2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1).unsqueeze(1)
                offset = torch.cat([offset_11, offset_12, offset_2], dim=0)

        elif qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union':
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1

                head_1 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)
                if self.euo and self.geo == 'box':
                    head_offset_1 = torch.index_select(self.entity_offset_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                    head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                    head_offset = torch.cat([head_offset_1, head_offset_2], dim=0)
                if rel_len == 3:
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)
                    if self.euo and self.geo == 'box':
                        head_offset_3 = torch.index_select(self.entity_offset_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                        head_offset = torch.cat([head_offset, head_offset_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:,-1]).unsqueeze(1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=sample[:,1]).unsqueeze(1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:,3]).unsqueeze(1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_embedding, dim=0, index=sample[:,5]).unsqueeze(1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                if self.geo == 'box':
                    offset_1 = torch.index_select(self.offset_embedding, dim=0, index=sample[:,1]).unsqueeze(1).unsqueeze(1)
                    offset_2 = torch.index_select(self.offset_embedding, dim=0, index=sample[:,3]).unsqueeze(1).unsqueeze(1)
                    offset = torch.cat([offset_1, offset_2], dim=0)
                    if rel_len == 3:
                        offset_3 = torch.index_select(self.offset_embedding, dim=0, index=sample[:,5]).unsqueeze(1).unsqueeze(1)
                        offset = torch.cat([offset, offset_3], dim=0)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                
                head_1 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                head = torch.cat([head_1, head_2], dim=0)
                if self.euo and self.geo == 'box':
                    head_offset_1 = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                    head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                    head_offset = torch.cat([head_offset_1, head_offset_2], dim=0)
                if rel_len == 3:
                    head_3 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)
                    head = torch.cat([head, head_3], dim=0)
                    if self.euo and self.geo == 'box':
                        head_offset_3 = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)
                        head_offset = torch.cat([head_offset, head_offset_3], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
                if rel_len == 2:
                    tail = torch.cat([tail, tail], dim=0)
                elif rel_len == 3:
                    tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1)
                relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1).unsqueeze(1)
                relation = torch.cat([relation_1, relation_2], dim=0)
                if rel_len == 3:
                    relation_3 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 5]).unsqueeze(1).unsqueeze(1)
                    relation = torch.cat([relation, relation_3], dim=0)

                if self.geo == 'box':
                    offset_1 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1)
                    offset_2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1).unsqueeze(1)
                    offset = torch.cat([offset_1, offset_2], dim=0)
                    if rel_len == 3:
                        offset_3 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 5]).unsqueeze(1).unsqueeze(1)
                        offset = torch.cat([offset, offset_3], dim=0)

        elif qtype == '1-chain' or qtype == '2-chain' or qtype == '3-chain':
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1
                
                head = torch.index_select(self.entity_embedding, dim=0, index=sample[:,0]).unsqueeze(1)
                
                relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:,1]).unsqueeze(1).unsqueeze(1)
                if self.geo == 'box':
                    offset = torch.index_select(self.offset_embedding, dim=0, index=sample[:,1]).unsqueeze(1).unsqueeze(1)
                    if self.euo:
                        head_offset = torch.index_select(self.entity_offset_embedding, dim=0, index=sample[:,0]).unsqueeze(1)
                if rel_len == 2 or rel_len == 3:
                    relation2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 2]).unsqueeze(1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)
                    if self.geo == 'box':
                        offset2 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 2]).unsqueeze(1).unsqueeze(1)
                        offset = torch.cat([offset, offset2], 1)
                if rel_len == 3:
                    relation3 = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 3]).unsqueeze(1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)
                    if self.geo == 'box':
                        offset3 = torch.index_select(self.offset_embedding, dim=0, index=sample[:, 3]).unsqueeze(1).unsqueeze(1)
                        offset = torch.cat([offset, offset3], 1)
                
                assert relation.size(1) == rel_len
                if self.geo == 'box':
                    assert offset.size(1) == rel_len
                    
                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:,-1]).unsqueeze(1)

            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                
                head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                
                relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1)
                if self.geo == 'box':
                    offset = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1)
                    if self.euo:
                        head_offset = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                if rel_len == 2 or rel_len == 3:
                    relation2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1).unsqueeze(1)
                    relation = torch.cat([relation, relation2], 1)
                    if self.geo == 'box':
                        offset2 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1).unsqueeze(1)
                        offset = torch.cat([offset, offset2], 1)
                if rel_len == 3:
                    relation3 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1).unsqueeze(1)
                    relation = torch.cat([relation, relation3], 1)
                    if self.geo == 'box':
                        offset3 = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1).unsqueeze(1)
                        offset = torch.cat([offset, offset3], 1)

                assert relation.size(1) == rel_len
                if self.geo == 'box':
                    assert offset.size(1) == rel_len
                
                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'BoxTransE': self.BoxTransE,
            'TransE': self.TransE,
        }
        if self.geo == 'vec':
            offset = None
            head_offset = None
        if self.geo == 'box':
            if not self.euo:
                head_offset = None
        
        if self.model_name in model_func:
            if qtype == '2-inter' or qtype == '3-inter' or qtype == '2-union' or qtype == '3-union':
                score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](head, relation, tail, mode, offset, head_offset, 1, qtype)
            else:
                score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](head, relation, tail, mode, offset, head_offset, rel_len, qtype)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score, score_cen, offset_norm, score_cen_plus, None, None
    
    def BoxTransE(self, head, relation, tail, mode, offset, head_offset, rel_len, qtype):

        if qtype == 'chain-inter':
            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            if self.euo:
                head_offsets = torch.chunk(head_offset, 2, dim=0)
            
            heads = torch.chunk(head, 2, dim=0)

            query_center_1 = heads[0] + relations[0][:,0,:,:] + relations[1][:,0,:,:]
            query_center_2 = heads[1] + relations[2][:,0,:,:]
            if self.euo:
                query_min_1 = query_center_1 - 0.5 * self.func(head_offsets[0]) - 0.5 * self.func(offsets[0][:,0,:,:]) - 0.5 * self.func(offsets[1][:,0,:,:])
                query_min_2 = query_center_2 - 0.5 * self.func(head_offsets[1]) - 0.5 * self.func(offsets[2][:,0,:,:])
                query_max_1 = query_center_1 + 0.5 * self.func(head_offsets[0]) + 0.5 * self.func(offsets[0][:,0,:,:]) + 0.5 * self.func(offsets[1][:,0,:,:])
                query_max_2 = query_center_2 + 0.5 * self.func(head_offsets[1]) + 0.5 * self.func(offsets[2][:,0,:,:])
            else:
                query_min_1 = query_center_1 - 0.5 * self.func(offsets[0][:,0,:,:]) - 0.5 * self.func(offsets[1][:,0,:,:])
                query_min_2 = query_center_2 - 0.5 * self.func(offsets[2][:,0,:,:])
                query_max_1 = query_center_1 + 0.5 * self.func(offsets[0][:,0,:,:]) + 0.5 * self.func(offsets[1][:,0,:,:])
                query_max_2 = query_center_2 + 0.5 * self.func(offsets[2][:,0,:,:])
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            offset_1 = (query_max_1 - query_min_1).squeeze(1)
            offset_2 = (query_max_2 - query_min_2).squeeze(1)
            new_query_center = self.center_sets(query_center_1, offset_1, query_center_2, offset_2)
            new_offset = self.offset_sets(query_center_1, offset_1, query_center_2, offset_2)
            new_query_min = (new_query_center - 0.5 * self.func(new_offset)).unsqueeze(1)
            new_query_max = (new_query_center + 0.5 * self.func(new_offset)).unsqueeze(1)
            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max)
            score_center = new_query_center.unsqueeze(1) - tail
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1)

        elif qtype == 'inter-chain':
            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            if self.euo:
                head_offsets = torch.chunk(head_offset, 2, dim=0)
            
            heads = torch.chunk(head, 2, dim=0)

            query_center_1 = heads[0] + relations[0][:,0,:,:]
            query_center_2 = heads[1] + relations[1][:,0,:,:]
            if self.euo:
                query_min_1 = query_center_1 - 0.5 * self.func(head_offsets[0]) - 0.5 * self.func(offsets[0][:,0,:,:])
                query_min_2 = query_center_2 - 0.5 * self.func(head_offsets[1]) - 0.5 * self.func(offsets[1][:,0,:,:])
                query_max_1 = query_center_1 + 0.5 * self.func(head_offsets[0]) + 0.5 * self.func(offsets[0][:,0,:,:])
                query_max_2 = query_center_2 + 0.5 * self.func(head_offsets[1]) + 0.5 * self.func(offsets[1][:,0,:,:])
            else:
                query_min_1 = query_center_1 - 0.5 * self.func(offsets[0][:,0,:,:])
                query_min_2 = query_center_2 - 0.5 * self.func(offsets[1][:,0,:,:])
                query_max_1 = query_center_1 + 0.5 * self.func(offsets[0][:,0,:,:])
                query_max_2 = query_center_2 + 0.5 * self.func(offsets[1][:,0,:,:])
            query_center_1 = query_center_1.squeeze(1)
            query_center_2 = query_center_2.squeeze(1)
            offset_1 = (query_max_1 - query_min_1).squeeze(1)
            offset_2 = (query_max_2 - query_min_2).squeeze(1)
            conj_query_center = self.center_sets(query_center_1, offset_1, query_center_2, offset_2).unsqueeze(1)
            new_query_center = conj_query_center + relations[2][:,0,:,:]
            new_offset = self.offset_sets(query_center_1, offset_1, query_center_2, offset_2).unsqueeze(1)
            new_query_min = new_query_center - 0.5 * self.func(new_offset) - 0.5 * self.func(offsets[2][:,0,:,:])
            new_query_max = new_query_center + 0.5 * self.func(new_offset) + 0.5 * self.func(offsets[2][:,0,:,:])
            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max)
            score_center = new_query_center - tail
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center

        elif qtype == 'union-chain':
            relations = torch.chunk(relation, 3, dim=0)
            offsets = torch.chunk(offset, 3, dim=0)
            if self.euo:
                head_offsets = torch.chunk(head_offset, 2, dim=0)
            
            heads = torch.chunk(head, 2, dim=0)

            query_center_1 = heads[0] + relations[0][:,0,:,:] + relations[2][:,0,:,:]
            query_center_2 = heads[1] + relations[1][:,0,:,:] + relations[2][:,0,:,:]
            if self.euo:
                query_min_1 = query_center_1 - 0.5 * self.func(head_offsets[0]) - 0.5 * self.func(offsets[0][:,0,:,:]) - 0.5 * self.func(offsets[2][:,0,:,:])
                query_min_2 = query_center_2 - 0.5 * self.func(head_offsets[1]) - 0.5 * self.func(offsets[1][:,0,:,:]) - 0.5 * self.func(offsets[2][:,0,:,:])
                query_max_1 = query_center_1 + 0.5 * self.func(head_offsets[0]) + 0.5 * self.func(offsets[0][:,0,:,:]) + 0.5 * self.func(offsets[2][:,0,:,:])
                query_max_2 = query_center_2 + 0.5 * self.func(head_offsets[1]) + 0.5 * self.func(offsets[1][:,0,:,:]) + 0.5 * self.func(offsets[2][:,0,:,:])
            else:
                query_min_1 = query_center_1 - 0.5 * self.func(offsets[0][:,0,:,:]) - 0.5 * self.func(offsets[2][:,0,:,:])
                query_min_2 = query_center_2 - 0.5 * self.func(offsets[1][:,0,:,:]) - 0.5 * self.func(offsets[2][:,0,:,:])
                query_max_1 = query_center_1 + 0.5 * self.func(offsets[0][:,0,:,:]) + 0.5 * self.func(offsets[2][:,0,:,:])
                query_max_2 = query_center_2 + 0.5 * self.func(offsets[1][:,0,:,:]) + 0.5 * self.func(offsets[2][:,0,:,:])

            new_query_min = torch.stack([query_min_1, query_min_2], dim=0)
            new_query_max = torch.stack([query_max_1, query_max_2], dim=0)
            new_query_center = torch.stack([query_center_1, query_center_2], dim=0)
            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max)
            score_center = new_query_center - tail
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center

        else:
            query_center = head
            for rel in range(rel_len):
                query_center = query_center + relation[:,rel,:,:]
            if self.euo:
                query_min = query_center - 0.5 * self.func(head_offset)
                query_max = query_center + 0.5 * self.func(head_offset)
            else:
                query_min = query_center
                query_max = query_center
            for rel in range(0, rel_len):
                query_min = query_min - 0.5 * self.func(offset[:,rel,:,:])
                query_max = query_max + 0.5 * self.func(offset[:,rel,:,:])

            if 'inter' not in qtype and 'union' not in qtype:
                score_offset = F.relu(query_min - tail) + F.relu(tail - query_max)
                score_center = query_center - tail
                score_center_plus = torch.min(query_max, torch.max(query_min, tail)) - query_center
            else:
                rel_len = int(qtype.split('-')[0])
                assert rel_len > 1
                queries_min = torch.chunk(query_min, rel_len, dim=0)
                queries_max = torch.chunk(query_max, rel_len, dim=0)
                queries_center = torch.chunk(query_center, rel_len, dim=0)
                tails = torch.chunk(tail, rel_len, dim=0)
                offsets = query_max - query_min
                offsets = torch.chunk(offsets, rel_len, dim=0)
                if 'inter' in qtype:
                    if rel_len == 2:
                        new_query_center = self.center_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1), 
                                                        queries_center[1].squeeze(1), offsets[1].squeeze(1))
                        new_offset = self.offset_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                        queries_center[1].squeeze(1), offsets[1].squeeze(1))
                        
                    elif rel_len == 3:
                        new_query_center = self.center_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1), 
                                                        queries_center[1].squeeze(1), offsets[1].squeeze(1), 
                                                        queries_center[2].squeeze(1), offsets[2].squeeze(1))
                        new_offset = self.offset_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                                                        queries_center[1].squeeze(1), offsets[1].squeeze(1),
                                                        queries_center[2].squeeze(1), offsets[2].squeeze(1))
                    new_query_min = (new_query_center - 0.5*self.func(new_offset)).unsqueeze(1)
                    new_query_max = (new_query_center + 0.5*self.func(new_offset)).unsqueeze(1)
                    score_offset = F.relu(new_query_min - tails[0]) + F.relu(tails[0] - new_query_max)
                    score_center = new_query_center.unsqueeze(1) - tails[0]
                    score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tails[0])) - new_query_center.unsqueeze(1)
                elif 'union' in qtype:
                    new_query_min = torch.stack(queries_min, dim=0)
                    new_query_max = torch.stack(queries_max, dim=0)
                    new_query_center = torch.stack(queries_center, dim=0)
                    score_offset = F.relu(new_query_min - tails[0]) + F.relu(tails[0] - new_query_max)
                    score_center = new_query_center - tails[0]
                    score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tails[0])) - new_query_center
                else:
                    assert False, 'qtype not exists: %s'%qtype
        score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
        score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
        score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
        if 'union' in qtype:
            score = torch.max(score, dim=0)[0]
            score_center = torch.max(score_center, dim=0)[0]
            score_center_plus = torch.max(score_center_plus, dim=0)[0]

        return score, score_center, torch.mean(torch.norm(offset, p=2, dim=2).squeeze(1)), score_center_plus, None
    
    def TransE(self, head, relation, tail, mode, offset, head_offset, rel_len, qtype):

        if qtype == 'chain-inter':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = (heads[0] + relations[0][:,0,:,:] + relations[1][:,0,:,:]).squeeze(1)
            score_2 = (heads[1] + relations[2][:,0,:,:]).squeeze(1)
            conj_score = self.deepsets(score_1, None, score_2, None).unsqueeze(1)
            score = conj_score - tail
        elif qtype == 'inter-chain':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = (heads[0] + relations[0][:,0,:,:]).squeeze(1)
            score_2 = (heads[1] + relations[1][:,0,:,:]).squeeze(1)
            conj_score = self.deepsets(score_1, None, score_2, None).unsqueeze(1)
            score = conj_score + relations[2][:,0,:,:] - tail
        elif qtype == 'union-chain':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = heads[0] + relations[0][:,0,:,:] + relations[2][:,0,:,:]
            score_2 = heads[1] + relations[1][:,0,:,:] + relations[2][:,0,:,:]
            conj_score = torch.stack([score_1, score_2], dim=0)
            score = conj_score - tail
        else:
            score = head
            for rel in range(rel_len):
                score = score + relation[:,rel,:,:]

            if 'inter' not in qtype and 'union' not in qtype:
                score = score - tail
            else:
                rel_len = int(qtype.split('-')[0])
                assert rel_len > 1
                score = score.squeeze(1)
                scores = torch.chunk(score, rel_len, dim=0)
                tails = torch.chunk(tail, rel_len, dim=0)
                if 'inter' in qtype:
                    if rel_len == 2:
                        conj_score = self.deepsets(scores[0], None, scores[1], None)
                    elif rel_len == 3:
                        conj_score = self.deepsets(scores[0], None, scores[1], None, scores[2], None)
                    conj_score = conj_score.unsqueeze(1)
                    score = conj_score - tails[0]
                elif 'union' in qtype:
                    conj_score = torch.stack(scores, dim=0)
                    score = conj_score - tails[0]    
                else:
                    assert False, 'qtype not exist: %s'%qtype                    
        
        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)
        if 'union' in qtype:
            score = torch.max(score, dim=0)[0]
        if qtype == '2-union':
            score = score.unsqueeze(0)
        return score, None, None, 0., []

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        rel_len = int(train_iterator.qtype.split('-')[0])
        qtype = train_iterator.qtype
        negative_score, negative_score_cen, negative_offset, negative_score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len, qtype, mode=mode)

        if model.geo == 'box':
            negative_score = F.logsigmoid(-negative_score_cen_plus).mean(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score, positive_score_cen, positive_offset, positive_score_cen_plus, _, _ = model(positive_sample, rel_len, qtype)
        if model.geo == 'box':
            positive_score = F.logsigmoid(positive_score_cen_plus).squeeze(dim = 1)
        else:
            positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()
            positive_sample_loss /= subsampling_weight.sum()
            negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()
        optimizer.step()
        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        return log
    
    @staticmethod
    def test_step(model, test_triples, test_ans, test_ans_hard, args):
        qtype = test_triples[0][-1]
        if qtype == 'chain-inter' or qtype == 'inter-chain' or qtype == 'union-chain':
            rel_len = 2
        else:
            rel_len = int(test_triples[0][-1].split('-')[0])
        
        model.eval()
        
        if qtype == 'inter-chain' or qtype == 'union-chain':
            test_dataloader_tail = DataLoader(
                TestInterChainDataset(
                    test_triples, 
                    test_ans, 
                    test_ans_hard,
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num), 
                collate_fn=TestDataset.collate_fn
            )
        elif qtype == 'chain-inter':
            test_dataloader_tail = DataLoader(
                TestChainInterDataset(
                    test_triples, 
                    test_ans, 
                    test_ans_hard,
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num), 
                collate_fn=TestDataset.collate_fn
            )
        elif 'inter' in qtype or 'union' in qtype:
            test_dataloader_tail = DataLoader(
                TestInterDataset(
                    test_triples, 
                    test_ans, 
                    test_ans_hard,
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num), 
                collate_fn=TestDataset.collate_fn
            )
        else:
            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    test_ans, 
                    test_ans_hard,
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num), 
                collate_fn=TestDataset.collate_fn
            )

        test_dataset_list = [test_dataloader_tail]
        # test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        logs = []

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode, query in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    assert batch_size == 1, batch_size

                    if 'inter' in qtype:
                        if model.geo == 'box':
                            _, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len, qtype, mode=mode)
                        else:
                            score, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len, qtype, mode=mode)
                    else:
                        score, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len, qtype, mode=mode)

                    if model.geo == 'box':
                        score = score_cen
                        score2 = score_cen_plus

                    score -= (torch.min(score) - 1)
                    ans = test_ans[query]
                    hard_ans = test_ans_hard[query]
                    all_idx = set(range(args.nentity))
                    false_ans = all_idx - ans
                    ans_list = list(ans)
                    hard_ans_list = list(hard_ans)
                    false_ans_list = list(false_ans)
                    ans_idxs = np.array(hard_ans_list)
                    vals = np.zeros((len(ans_idxs), args.nentity))
                    vals[np.arange(len(ans_idxs)), ans_idxs] = 1
                    axis2 = np.tile(false_ans_list, len(ans_idxs))
                    axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))
                    vals[axis1, axis2] = 1
                    b = torch.Tensor(vals) if not args.cuda else torch.Tensor(vals).cuda()
                    filter_score = b*score
                    argsort = torch.argsort(filter_score, dim=1, descending=True)
                    ans_tensor = torch.LongTensor(hard_ans_list) if not args.cuda else torch.LongTensor(hard_ans_list).cuda()
                    argsort = torch.transpose(torch.transpose(argsort, 0, 1) - ans_tensor, 0, 1)
                    ranking = (argsort == 0).nonzero()
                    ranking = ranking[:, 1]
                    ranking = ranking + 1
                    if model.geo == 'box':
                        score2 -= (torch.min(score2) - 1)
                        filter_score2 = b*score2
                        argsort2 = torch.argsort(filter_score2, dim=1, descending=True)
                        argsort2 = torch.transpose(torch.transpose(argsort2, 0, 1) - ans_tensor, 0, 1)
                        ranking2 = (argsort2 == 0).nonzero()
                        ranking2 = ranking2[:, 1]
                        ranking2 = ranking2 + 1

                    ans_vec = np.zeros(args.nentity)
                    ans_vec[ans_list] = 1
                    hits1 = torch.sum((ranking <= 1).to(torch.float)).item()
                    hits3 = torch.sum((ranking <= 3).to(torch.float)).item()
                    hits10 = torch.sum((ranking <= 10).to(torch.float)).item()
                    mr = float(torch.sum(ranking).item())
                    mrr = torch.sum(1./ranking.to(torch.float)).item()
                    hits1m = torch.mean((ranking <= 1).to(torch.float)).item()
                    hits3m = torch.mean((ranking <= 3).to(torch.float)).item()
                    hits10m = torch.mean((ranking <= 10).to(torch.float)).item()
                    mrm = torch.mean(ranking.to(torch.float)).item()
                    mrrm = torch.mean(1./ranking.to(torch.float)).item()
                    num_ans = len(hard_ans_list)
                    if model.geo == 'box':
                        hits1m_newd = torch.mean((ranking2 <= 1).to(torch.float)).item()
                        hits3m_newd = torch.mean((ranking2 <= 3).to(torch.float)).item()
                        hits10m_newd = torch.mean((ranking2 <= 10).to(torch.float)).item()
                        mrm_newd = torch.mean(ranking2.to(torch.float)).item()
                        mrrm_newd = torch.mean(1./ranking2.to(torch.float)).item()
                    else:
                        hits1m_newd = hits1m
                        hits3m_newd = hits3m
                        hits10m_newd = hits10m
                        mrm_newd = mrm
                        mrrm_newd = mrrm

                    logs.append({
                        'MRRm_new': mrrm_newd,
                        'MRm_new': mrm_newd,
                        'HITS@1m_new': hits1m_newd,
                        'HITS@3m_new': hits3m_newd,
                        'HITS@10m_new': hits10m_newd,
                        'num_answer': num_ans
                    })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        num_answer = sum([log['num_answer'] for log in logs])
        for metric in logs[0].keys():
            if metric == 'num_answer':
                continue
            if 'm' in metric:
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)
            else:
                metrics[metric] = sum([log[metric] for log in logs])/num_answer
        return metrics