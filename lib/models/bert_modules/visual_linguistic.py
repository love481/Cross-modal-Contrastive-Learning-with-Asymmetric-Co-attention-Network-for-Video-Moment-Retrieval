from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import math
from .vtc import VTC
from .modeling_mplug import BertLayerNorm, FusionEncoder

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position = 116): ## d_hid defines the size of embedding
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i / n_position) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2] * 2 * math.pi)  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2] * 2 * math.pi)  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        self.config = config
        super(BaseModel, self).__init__()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, *args, **kwargs):
        raise NotImplemented


class VisualLinguisticTranformer(BaseModel):
    def __init__(self, dataset, config):
        super(VisualLinguisticTranformer, self).__init__(config)

        self.config = config

        # embeddings
        self.mask_embeddings = nn.Embedding(1, config.hidden_size) 
        self.word_mapping = nn.Linear(300, config.hidden_size)    # 300 is the dim of glove vector
        self.mplug = VTC(config)  # initialize the Video Text Contrastive Loss module
        self.text_embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.text_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.visual_embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.visual_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        if dataset == "ActivityNet":
            self.postion_encoding = PositionalEncoding(config.hidden_size, n_position=116)
        elif dataset == "TACoS":
            self.postion_encoding = PositionalEncoding(config.hidden_size, n_position=194)
        else:
            print('DATASET ERROR')
            exit()

        # Skip Connected Network Containing Asymmetric co-attention Network
        self.encoder=FusionEncoder(config)
        self.visual_1x1_object = None
        if config.visual_size != config.hidden_size:
            self.visual_1x1_visual =nn.Linear(config.visual_size, config.hidden_size)


        # init weights
        self.apply(self.init_weights)

    def forward(self,
                text_input_feats,
                text_mask,
                word_mask,
                object_visual_embeddings,
                anno_idxs,gt_times):

        # get seamless concatenate embeddings and mask
        ## now the text_embeddings and the visual embeddings are all combined with the 
        # positional embeddings and they are later layer normalized and the dropout is added
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = text_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        visual_attention_mask = torch.zeros(( object_visual_embeddings.shape[0],1, 1, object_visual_embeddings.shape[1])).float().to(object_visual_embeddings.device)
        text_embeddings, visual_embeddings = self.embedding(text_input_feats, word_mask,object_visual_embeddings)
        visual_embeddings,text_embeddings,loss_itc = self.mplug(text_embeddings, visual_embeddings ,self.config.alpha, anno_idxs,gt_times)
        encoded_layers = self.encoder(text_embeddings, attention_mask = extended_attention_mask,encoder_hidden_states = visual_embeddings, encoder_attention_mask = visual_attention_mask)
        visual_embeddings, text_embeddings = encoded_layers
        return text_embeddings, visual_embeddings, loss_itc

    def embedding(self,
                  text_input_feats,
                  word_mask,
                  object_visual_embeddings):
        text_linguistic_embedding = self.word_mapping(text_input_feats) ## convert embedding dimension into hidden_size
        text_input_feats_temp = text_linguistic_embedding.clone()
        if self.training:
            _zero_id = torch.zeros(text_input_feats_temp.shape[:2], dtype=torch.long, device= text_input_feats_temp.device)
            text_input_feats_temp[word_mask>0] = self.mask_embeddings(_zero_id)[word_mask>0] ##convert the masked word into mask embedding
        if self.visual_1x1_visual is not None:
            object_visual_embeddings = self.visual_1x1_visual(object_visual_embeddings) ## convert embedding dimension into hidden_size

        embeddings = torch.cat([object_visual_embeddings, text_input_feats_temp], dim=1) ## concatenate the visual and text embeddings
        embeddings = self.postion_encoding(embeddings)  ## demension of position embedding is same as feature dimensions for both visual and text
        visual_embeddings, text_embeddings = torch.split(embeddings, [object_visual_embeddings.size(1),text_input_feats_temp.size(1)], 1)
        text_embeddings = self.text_embedding_LayerNorm(text_embeddings)
        text_embeddings = self.text_embedding_dropout(text_embeddings)

        visual_embeddings = self.visual_embedding_LayerNorm(visual_embeddings)
        visual_embeddings = self.visual_embedding_dropout(visual_embeddings)
        return text_embeddings, visual_embeddings



