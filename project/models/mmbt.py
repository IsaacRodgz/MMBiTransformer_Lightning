#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME
from collections import OrderedDict

from models.image import ImageEncoder


class ImageBertEmbeddings(nn.Module):
    def __init__(self, hparams, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.hparams = hparams
        self.img_embeddings = nn.Linear(hparams.img_hidden_sz, hparams.hidden_sz)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=hparams.dropout)

    def forward(self, input_imgs, token_type_ids):
        bsz = input_imgs.size(0)
        seq_length = self.hparams.num_image_embeds + 2  # +2 for CLS and SEP Token

        cls_id = torch.LongTensor([self.hparams.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.hparams.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)

        imgs_embeddings = self.img_embeddings(input_imgs)
        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1
        )

        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultimodalBertEncoder(nn.Module):
    def __init__(self, hparams):
        super(MultimodalBertEncoder, self).__init__()
        self.hparams = hparams

        bert = BertModel.from_pretrained(hparams.bert_model)
        self.txt_embeddings = bert.embeddings

        if hparams.task == "vsnli":
            ternary_embeds = nn.Embedding(3, hparams.hidden_sz)
            ternary_embeds.weight.data[:2].copy_(
                bert.embeddings.token_type_embeddings.weight
            )
            ternary_embeds.weight.data[2].copy_(
                bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)
            )
            self.txt_embeddings.token_type_embeddings = ternary_embeds
            
        if self.hparams.pooling == 'cls_att':
            pooling_dim = 2*hparams.hidden_sz
        else:
            pooling_dim = hparams.hidden_sz

        self.img_embeddings = ImageBertEmbeddings(hparams, self.txt_embeddings)
        self.img_encoder = ImageEncoder(hparams)
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.pooler_custom = nn.Sequential(
          nn.Linear(pooling_dim, hparams.hidden_sz),
          nn.Tanh(),
        )
        self.att_query = nn.Parameter(torch.rand(hparams.hidden_sz))
        self.clf = nn.Linear(hparams.hidden_sz, hparams.n_classes)

    def forward(self, input_txt, attention_mask, segment, input_img):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, self.hparams.num_image_embeds + 2).long().cuda(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        img_tok = (
            torch.LongTensor(input_txt.size(0), self.hparams.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )
        img = self.img_encoder(input_img)  # BxNx3x224x224 -> BxNx2048
        img_embed_out = self.img_embeddings(img, img_tok)
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID
        
        # Output all encoded layers only for vertical attention on CLS token
        encoded_layers = self.encoder(
                encoder_input, extended_attention_mask, output_all_encoded_layers=(self.hparams.pooling == 'vert_att')
            )
        
        if self.hparams.pooling == 'cls':
            output = self.pooler(encoded_layers[-1])
        
        elif self.hparams.pooling == 'att':
            hidden = encoded_layers[-1]  # Get all hidden vectors of last layer (B, L, hidden_sz)
            dot = (hidden*self.att_query).sum(-1)  # Matrix of dot products (B, L)
            weights = F.softmax(dot, dim=1).unsqueeze(2)  # Normalize dot products and expand last dim (B, L, 1)
            weighted_sum = (hidden*weights).sum(dim=1)  # Weighted sum of hidden vectors (B, hidden_sz)
            output = self.pooler_custom(weighted_sum)
            
        elif self.hparams.pooling == 'cls_att':
            hidden = encoded_layers[-1]  # Get all hidden vectors of last layer (B, L, hidden_sz)
            cls_token = hidden[:, 0]  # Extract vector of CLS token
            word_tokens = hidden[:, 1:]
            dot = (word_tokens*self.att_query).sum(-1)  # Matrix of dot products (B, L)
            weights = F.softmax(dot, dim=1).unsqueeze(2)  # Normalize dot products and expand last dim (B, L, 1)
            weighted_sum = (word_tokens*weights).sum(dim=1)  # Weighted sum of hidden vectors (B, hidden_sz)
            pooler_cat = torch.cat([cls_token, weighted_sum], dim=1)
            output = self.pooler_custom(pooler_cat)
        
        else:
            hidden = [cls_hidden[:, 0] for cls_hidden in encoded_layers]  # Get all hidden vectors corresponding to CLS token (B, Num_bert_layers, hidden_sz)
            hidden = torch.stack(hidden, dim=1)  # Convert to tensor (B, Num_bert_layers, hidden_sz)
            dot = (hidden*self.att_query).sum(-1)  # Matrix of dot products (B, Num_bert_layers)
            weights = F.softmax(dot, dim=1).unsqueeze(2)  # Normalize dot products and expand last dim (B, Num_bert_layers, 1)
            weighted_sum = (hidden*weights).sum(dim=1)  # Weighted sum of hidden vectors (B, hidden_sz)
            output = self.pooler_custom(weighted_sum)

        return output


class MultimodalBertClf(nn.Module):
    def __init__(self, hparams):
        super(MultimodalBertClf, self).__init__()
        self.hparams = hparams

        self.enc = MultimodalBertEncoder(hparams)
        self.clf = nn.Linear(hparams.hidden_sz, hparams.n_classes)

    def forward(self, txt, mask, segment, img):
        x = self.enc(txt, mask, segment, img)
        return self.clf(x)
