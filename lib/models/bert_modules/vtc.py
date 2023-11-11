from .modeling_mplug import FusionEncoder
import torch
from torch import nn
import torch.nn.functional as F
class VTC(nn.Module):
    def __init__(self,config=None,):
        super().__init__()
        self.module_setting(config)
        self.visual_encoder=  nn.Linear(config.hidden_size, config.hidden_size)
        self.text_encoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_distill(config)

    def forward(self,text, image, alpha, idx,gt_times):
        image_embeds = self.visual_encoder(image)
        text_embeds  = self.text_encoder(text)
        image_pos_clip=torch.zeros((image_embeds.shape[0], image_embeds.shape[2])).cuda()
        if self.training:
            for i in range(0,image.shape[0]):
                image_pos_clip[i,:]=torch.sum(image_embeds[i,torch.ceil(gt_times[i,0]).long():torch.ceil(gt_times[i,1]).long(),:],dim=0)

            image_feat = F.normalize(self.vision_proj(image_pos_clip), dim=-1)
            text_feat = F.normalize(self.text_proj(torch.sum(text_embeds,dim=1)), dim=-1)

            idx = idx.view(-1, 1)
            idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(image)
                image_pos_clip_m=torch.zeros((image_embeds_m.shape[0], image_embeds_m.shape[2])).cuda()
                for i in range(0,image.shape[0]):
                    image_pos_clip_m[i,:]=torch.sum(image_embeds_m[i,torch.ceil(gt_times[i,0]).long():torch.ceil(gt_times[i,1]).long(),:],dim=0)
                image_feat_m = F.normalize(self.vision_proj_m(image_pos_clip_m), dim=-1)
                image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
                text_output_m = self.text_encoder_m(text)
                text_feat_m = F.normalize(self.text_proj_m(torch.sum(text_output_m,dim=1)), dim=-1)
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

                if self.distill:
                    sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                    sim_t2i_m = text_feat_m @ image_feat_all / self.temp

                    sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                    sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp

            if self.distill:
                loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
                loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            else:
                loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
                loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

            loss_itc = (loss_i2t + loss_t2i) / 2
            self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
        else:
            loss_itc = torch.zeros(1).cuda()
        return  image_embeds,text_embeds,loss_itc

    def module_setting(self, config):
        self.embed_dim = config.hidden_size
        self.temp = nn.Parameter(torch.ones([]) * config.temp)
        self.queue_size = config.queue_size
        self.momentum = config.momentum
        self.text_width = config.hidden_size

        self.vision_proj = nn.Linear(config.hidden_size, self.embed_dim)
        self.text_proj = nn.Linear(config.hidden_size, self.embed_dim)
        self.itm_head = nn.Linear(config.hidden_size, 2)

        self.register_buffer("image_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def init_distill(self, config):
        self.distill = config.distill
        if self.distill:
            self.visual_encoder_m =  nn.Linear(config.hidden_size, config.hidden_size)
            self.text_encoder_m = nn.Linear(config.hidden_size, config.hidden_size)
            self.vision_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.text_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_proj, self.text_proj_m],
                                [self.vision_proj, self.vision_proj_m],
                               ]
            self.copy_params()
            self.momentum = config.momentum

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats, idxs):
        # gather keys before updating queue
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)

        ##implemented as a circular queue
        indexx=torch.arange(ptr,ptr + batch_size)%self.queue_size
        self.image_queue[:, indexx] = image_feats.T
        self.text_queue[:, indexx] = text_feats.T
        self.idx_queue[:, indexx] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr
