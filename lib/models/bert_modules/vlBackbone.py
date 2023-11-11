import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .visual_linguistic import VisualLinguisticTranformer

BERT_WEIGHTS_NAME = 'pytorch_model.bin'
##TO DO find whether sigmoid is used or not

class mPLUG(nn.Module):
    def __init__(self, dataset, config):

        super(mPLUG, self).__init__()

        self.config = config
        if dataset ==  "ActivityNet":
            iou_mask_map = torch.zeros(33,33).float()
            for i in range(0,32,1):
                iou_mask_map[i,i+1:min(i+17,33)] = 1.
            for i in range(0,32-16,2):
                iou_mask_map[i,range(18+i,33,2)] = 1.
        elif dataset ==  "TACoS":
            iou_mask_map = torch.zeros(129,129).float()
            for i in range(0,128,1):
                iou_mask_map[i,1+i:min(i+17,129)] = 1.
            for i in range(0,128-16,2):
                iou_mask_map[i,range(18+i,min(33+i,129),2)] = 1.
            for i in range(0,128-32,4):
                iou_mask_map[i,range(36+i,min(65+i,129),4)] = 1.
            for i in range(0,128-64,8):
                iou_mask_map[i,range(72+i,129,8)] = 1.
        else:
            print('DATASET ERROR')
            exit()
            
        self.register_buffer('iou_mask_map', iou_mask_map)

        self.vlbert = VisualLinguisticTranformer(dataset, config)

        dim = config.hidden_size
        if config.CLASSIFIER_TYPE == "2fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, config.vocab_size)
            )
            self.final_mlp_2 = torch.nn.Sequential(
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, dim*3),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
            )
            self.final_mlp_3 = torch.nn.Sequential(
                torch.nn.Linear(dim*3, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 3)
            )
            self.final_mlp_s = torch.nn.Sequential(
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 1)
            )
            self.final_mlp_e = torch.nn.Sequential(
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 1)
            )
            self.final_mlp_c = torch.nn.Sequential(
                torch.nn.Linear(dim, config.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.CLASSIFIER_HIDDEN_SIZE, 1)
            )
        else:
            raise ValueError("Not support classifier type: {}!".format(config.CLASSIFIER_TYPE))

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_2.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_3.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def fix_params(self):
        pass


    def forward(self, text_input_feats, text_mask, word_mask, object_visual_feats, anno_idxs,gt_times):
        ###########################################

        # Visual Linguistic BERT
        #batch_size*no_words*512     And batch_size*129*512 
        hidden_states_text, hidden_states_object,loss_itc = self.vlbert(text_input_feats,
                                                                  text_mask,
                                                                  word_mask,
                                                                  object_visual_feats, anno_idxs,gt_times)
        
        logits_text = self.final_mlp(hidden_states_text)#batch_size*no_words*1514
        hidden_states_object = self.final_mlp_2(hidden_states_object) #torch.Size([4, 129, 1536])
        hidden_s, hidden_e, hidden_c = torch.split(hidden_states_object, self.config.hidden_size, dim=-1) ##define the  visual feature for respective frame

        T = hidden_states_object.size(1)
        s_idx = torch.arange(T, device=hidden_states_object.device)
        e_idx = torch.arange(T, device=hidden_states_object.device)
        c_point = hidden_c[:,(0.5*(s_idx[:,None] + e_idx[None,:])).long().flatten(),:].view(hidden_c.size(0),T,T,hidden_c.size(-1)) ##torch.Size([4, 129, 129, 512])
        s_c_e_points = torch.cat((hidden_s[:,:,None,:].repeat(1,1,T,1), c_point, hidden_e[:,None,:,:].repeat(1,T,1,1)), -1) ##torch.Size([4, 129, 129, 1536]) 
        logits_iou = self.final_mlp_3(s_c_e_points).permute(0,3,1,2).contiguous() ##torch.Size([4, 3, 129, 129])
        logits_visual = torch.cat((self.final_mlp_s(hidden_s), self.final_mlp_e(hidden_e), self.final_mlp_c(hidden_c)), -1) ##torch.Size([4, 129, 3]) define predictions value
        # logits_visual = logits_visual.permute(0,2,1).contiguous()

        return logits_text, logits_visual, logits_iou, self.iou_mask_map.clone().detach(),loss_itc
