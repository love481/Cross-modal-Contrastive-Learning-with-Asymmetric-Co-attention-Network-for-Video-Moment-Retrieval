import torch
import json
import numpy as np
import torchtext
from torch import nn
import torch.nn.functional as F
from torch import multiprocessing
from prettytable import PrettyTable
multiprocessing.set_sharing_strategy('file_system')
import train
from core.config import config
from datasets import average_to_fixed_length
import models
from collections import OrderedDict
import h5py
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    train_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        train_params+=param
    print(table)
    print(f"Total Trainable Params: {train_params}")



def iou(pred, gt): # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt,list)
    pred_is_list = isinstance(pred[0],list)
    gt_is_list = isinstance(gt[0],list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:,0,None], gt[None,:,0])
    inter_right = np.minimum(pred[:,1,None], gt[None,:,1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:,0,None], gt[None,:,0])
    union_right = np.maximum(pred[:,1,None], gt[None,:,1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:,0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap

def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]

def get_proposal_results(scores, regress, durations):
        out_sorted_times = []
        T = scores.shape[-1]

        regress = regress.cpu().detach().numpy()

        for score, reg, duration in zip(scores, regress, durations):
            sorted_indexs = np.dstack(np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array([[reg[0,item[0],item[1]], reg[1,item[0],item[1]]] for item in sorted_indexs[0] if reg[0,item[0],item[1]] < reg[1,item[0],item[1]] and score[item[0],item[1]].cpu().detach().numpy()>0.3]).astype(float)
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size =  config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            out_sorted_times.append((sorted_indexs.float() / target_size * duration).tolist())

        return out_sorted_times

class activitynet_single_data(torch.utils.data.Dataset):
    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self,video_name,sentence,gt_times,duration):

        super(activitynet_single_data, self).__init__()

        with open('./data/ActivityNet/words_vocab_activitynet.json', 'r') as f:
            tmp = json.load(f)
            self.itos = tmp['words']
        self.stoi = OrderedDict()
        for i, w in enumerate(self.itos):
            self.stoi[w] = i
        self.num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
        self.annos = None
        self._compute_annotations(video_name, sentence, gt_times, duration)


    def get_spec(self):
        anno = self.annos
        vid = anno['vid']
        query = self._get_language_feature(anno)
        visual_input, _ = self.get_video_features(vid)
        feat = average_to_fixed_length(visual_input)
        moment = anno['moment']
        # video features, input_sentence_embeddings, attentions_masks_with_all_1, No masking, target moment
        return feat, query[0], query[1], query[2], moment

    def __len__(self):
        return len(self.annos)

    def get_duration(self):
        return self.annos['duration']

    def get_sentence(self):
        return self.annos['sentence']

    def get_moment(self):
        return self.annos['moment']

    def get_vid(self):
        return self.annos['vid']

    def _compute_annotations(self,video_name,sentence,timestamp,duration):
        sentence = sentence.replace(',',' ').replace('/',' ').replace('\"',' ').replace('-',' ').replace(';',' ').replace('.',' ').replace('&',' ').replace('?',' ').replace('!',' ').replace('(',' ').replace(')',' ')
        word_label = [self.stoi.get(w.lower(), 10727) for w in sentence.split()]
        word_label = torch.tensor(word_label, dtype=torch.long)
        range_i = range(len(word_label))
        word_mask = [0. for _ in range_i]
        word_mask = torch.tensor(word_mask, dtype=torch.float)
        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        inps = self.word_embedding(word_idxs)
        attention_mask =  torch.ones(inps.shape[0],)
        query = (inps, attention_mask, word_mask)

        # moments changed to [0,32] range
        if timestamp[0] < timestamp[1]:
            moment = torch.tensor(
                [max(timestamp[0],0),
                    min(timestamp[1],duration)]
            )
        self.annos={
            'vid': video_name,
            'moment': moment,
            'sentence': sentence,
            'query': query,
            'duration': duration,
        }

    def _get_language_feature(self, anno):
        query = anno['query']
        return query

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
        with h5py.File('./data/ActivityNet/activitynet_v1-3_c3d.hdf5','r') as f:
            features = torch.from_numpy(f[vid]['c3d_features'][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

def inference(t,n_f,vid,sen):
    args = train.parse_args()
    train.reset_config(config, args)

    # loading model
    model_name = config.MODEL.NAME
    model = getattr(models, model_name)()
    count_parameters(model)
    model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
    model.load_state_dict(model_checkpoint)
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)
    model.eval()
    data_eg = activitynet_single_data(vid,sen,t, n_f)
    feats, inps, attention_masks, token_masks, moments = data_eg.get_spec()
    feats = feats.float().unsqueeze(0).to(device)
    inps = inps.unsqueeze(0).to(device)
    attention_masks = attention_masks.unsqueeze(0).to(device)
    token_masks=token_masks.unsqueeze(0).to(device)
    _, _, logits_iou, _, _ = model(inps, attention_masks, token_masks, feats, None, moments)
    T = logits_iou.shape[-1]
    idxs = torch.arange(T, device=logits_iou.device)
    s_e_idx = torch.cat((idxs[None,None,:T,None].repeat(1,1,1,T), idxs[None,None,None,:].repeat(1,1,T,1)), 1)
    regress = (s_e_idx + logits_iou[:,:2,:,:]).clone().detach()
    regress[:,1,:,:] = torch.min(torch.tensor(data_eg.num_clips).float().to(regress.device),  regress[:,1,:,:])
    regress[:,0,:,:] = torch.max(torch.tensor(0).float().to(regress.device),  regress[:,0,:,:])
    iou_scores = torch.sigmoid(logits_iou[:,2,:,:])
    sorted_times = get_proposal_results(iou_scores,regress,[data_eg.get_duration()])
    Total_predicted_moments = 1
    seg = nms(sorted_times[0], thresh=config.TEST.NMS_THRESH, top_k=Total_predicted_moments)
    if len(seg) == 0:
        seg = np.zeros([Total_predicted_moments,2]).tolist()
    print("Target: {}, Prediction: {}".format(moments,seg))
def get_video_feature(vid,sentence):
    with open('./data/ActivityNet/test.json','r') as f:
        annotations = json.load(f)
    return annotations[vid]['sentences'].index(sentence), annotations
if __name__=='__main__':
    vid = "v_bwRsZtPzipc"
    sen = "Two belly dancers with candles in each hand walk onto a dance floor with a mirrored wall."
    i, ann=get_video_feature(vid,sen)
    t = ann[vid]['timestamps'][i]
    n_f = ann[vid]['duration']
    inference(t, n_f, vid, sen)















