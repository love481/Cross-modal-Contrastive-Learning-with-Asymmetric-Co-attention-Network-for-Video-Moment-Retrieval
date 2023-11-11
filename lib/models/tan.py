import torch
from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.bert_modules as bert_modules

class TAN(nn.Module):
    def __init__(self):
        super(TAN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.bert_layer = getattr(bert_modules, config.TAN.VL_MODULE.NAME)(config.DATASET.NAME, config.TAN.VL_MODULE.PARAMS)

    def forward(self, textual_input, textual_mask, word_mask, visual_input,anno_idxs,gt_times):
        #print(textual_input, textual_mask, word_mask, visual_input,end='\n')
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        vis_h = vis_h.transpose(1, 2)   ## convert to batch_size * 129 * 4096 using avergaing two consecutive visual frame from 256
        logits_text, logits_visual, logits_iou, iou_mask_map,loss_itc = self.bert_layer(textual_input, textual_mask, word_mask, vis_h,anno_idxs,gt_times)
        # logits_text = logits_text.transpose(1, 2)
        logits_visual = logits_visual.transpose(1, 2) ##convert to batch_size * 3 * 129

        return logits_text, logits_visual, logits_iou, iou_mask_map, loss_itc

