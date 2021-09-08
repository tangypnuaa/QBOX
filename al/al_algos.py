import torch
import pickle
import copy
import numpy as np
from utils.utils import non_max_suppression_and_remove_overlap_with_gt
from models_da import create_grids
import torch.nn.functional as F
import torch.nn as nn
from utils.utils import non_max_suppression, bbox_iou

"""
inf_out: [1, 10647, 25] (batch, anchor boxes, predictions)
train_out: 3-tuple, with shape:
[1, 3, 13, 13, 25]
[1, 3, 26, 26, 25]
[1, 3, 52, 52, 25]
"""


def our_scoring_anchor_boxes(model_output, ins_da_output, img_da_output,
                             pos_ins_weight=0.05, da_tradeoff=0.5):
    """

    :param model_output: Tensor format: batch, boxes_ind, grid_x, grid_y, boxes_coord(4)+conf+class
    :param ins_da_output: Tensor format: batch, predict(2), grid_x, grid_y
    :param pos_ins_weight: weight for positive instance. the importance for the neg instance.
    :return:
    """
    inconsistency, transferrability, final_scores = [], [], []
    scale_num = len(model_output)
    for i in range(scale_num):
        conf = torch.sigmoid(model_output[i][..., 4])
        # weight = torch.ones_like(conf)
        # weight[conf < 0.5] = pos_ins_weight
        unc_score = uncertainty_scoring_anchor_boxes(model_output[i])
        inconsistency.append((conf ** pos_ins_weight) * unc_score)
        img_trans = img_trans_score(img_da_output, normalize=False)
        ins_trans = da_scoring_anchor_boxes(ins_da_output[i])
        transferrability.append(ins_trans*img_trans)
        final_score = inconsistency[i] + da_tradeoff * transferrability[i].repeat(1, inconsistency[i].shape[1], 1, 1)
        final_scores.append(final_score)
    return inconsistency, transferrability, final_scores


def uncertainty_scoring_anchor_boxes(model_output):
    bbxyp = model_output.shape
    unc = torch.zeros_like(model_output[..., 0])
    # calc bvsb for each anchor box and store the result in unc tensor
    conf = torch.sigmoid(model_output[..., 4])
    cls = torch.sigmoid(model_output[..., 5:])
    tp = torch.argsort(cls)
    scalars = torch.arange(bbxyp[2] * bbxyp[3] * bbxyp[1])*(bbxyp[4]-5)
    fst = tp[..., -1]
    scalars = scalars.reshape_as(fst).to(fst.device)
    fbv = torch.take(cls, scalars+fst)
    unc = (fbv - conf)**2
    return unc


def da_scoring_anchor_boxes(da_output):
    """Select instance depend on the transferrability only."""
    base_prob1 = torch.sigmoid(da_output)
    # da_score1 = (1-base_prob1[:, 1, ...]) / base_prob1[:, 1, ...]
    da_score1 = (1-base_prob1[:, 1, ...])
    return da_score1.reshape((1, da_score1.shape[0], da_score1.shape[1], da_score1.shape[2]))


def img_trans_score(img_da_output, normalize:'bool'=True):
    """Query example by image transferrability only."""
    # base_prob1 = F.softmax(img_da_output, dim=1)  # 1,2,13,13
    base_prob1 = torch.sigmoid(img_da_output)
    # loss is large -> predict as target domain, preferred.
    # loss is small -> predict as source domain
    # lab = torch.zeros((img_da_output.shape[0], img_da_output.shape[2], img_da_output.shape[3])).to(img_da_output.device)
    # total_da_loss = F.nll_loss(base_prob1, lab, reduction='mean')
    if normalize:
        return torch.mean((1 - base_prob1[:, 1, ...]) / base_prob1[:, 1, ...])  # source domain prob
    else:
        return torch.mean(1 - base_prob1[:, 1, ...])


def get_candidate_boxes(model_output, output_scores, model, gt_boxes,
                        score_thres=0.1, nms_thres=0.5, img_size=416,
                        store_part=False, part_array=None, min_area=15):
    """Run NMS for anchor boxes with al scores and remove boxes overlaps with GT.
    Also filter out boxes whose width or height less than 10 px.

    :param model_output: Tensor format: batch, boxes_ind, grid_x, grid_y, boxes_coord(xywh)+conf+class
    :param output_scores: Tensor format: batch, boxes_ind, grid_x, grid_y, scores
    :param model: nn.Module OD model
    :param gt_boxes: torch.Tensor, shape: (n_boxes, (image_id class xywh))
    :param score_thres: threshold to filter out low score instances
    :param nms_thres: threshold to filter out overlapped boxes
    :return:
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_conf, class)
    """
    # compute scores for each box.
    # NMS filter out low scored & overlapped boxes
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    tp = []
    for i in range(len(model_output)):
        tp.append(model_output[i].clone().detach().cpu())
        # replace conf and cls scores with al score for NMS
        tp[i][..., 4] = output_scores[i]
        tp[i][..., 5:] = torch.ones(tp[i].shape[-1]-5)
        if store_part:
            assert part_array is not None
            tp[i][..., 5] = part_array[i]

        # get number of grid points and anchor vec for this yolo layer
        if multi_gpu:
            try:
                ng, anchor_vec, anchors = model.module.pure_yolo.module_list[model.yolo_layers[i]].ng, \
                                          model.module.pure_yolo.module_list[
                                              model.yolo_layers[i]].anchor_vec, model.module.pure_yolo.module_list[
                                              model.yolo_layers[i]].anchors
            except AttributeError:
                ng, anchor_vec, anchors = model.module.module_list[model.yolo_layers[i]].ng, model.module.module_list[
                    model.yolo_layers[i]].anchor_vec, model.module.module_list[model.yolo_layers[i]].anchors
        else:
            try:
                ng, anchor_vec, anchors = model.pure_yolo.module_list[model.yolo_layers[i]].ng, \
                                          model.pure_yolo.module_list[model.yolo_layers[i]].anchor_vec, \
                                          model.pure_yolo.module_list[model.yolo_layers[i]].anchors
            except AttributeError:
                ng, anchor_vec, anchors = model.module_list[model.yolo_layers[i]].ng, model.module_list[
                    model.yolo_layers[i]].anchor_vec, model.module_list[
                                              model.yolo_layers[i]].anchors
        # convert to NMS format
        # different nums in different scales.
        # calc vars
        nx, ny = ng.to("cpu")  # x and y grid size
        na = len(anchors)
        stride = img_size / max(ng)

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid_xy = torch.stack((xv, yv), 2).view((1, 1, int(ny), int(nx), 2)).to("cpu")
        # build wh gains
        anchor_wh = anchor_vec.view(1, na, 1, 1, 2).to("cpu")

        tp[i][..., 0:2] = torch.sigmoid(tp[i][..., 0:2]) + grid_xy  # xy
        tp[i][..., 2:4] = torch.exp(tp[i][..., 2:4]) * anchor_wh  # wh yolo method
        # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
        tp[i][..., :4] *= stride
        # torch.sigmoid_(tp[i][..., 4:])    # no need to sigmoid again
        tp[i] = tp[i].view(1, -1, model_output[i].shape[-1])

    # concat tp
    pred_anchor_boxes = torch.cat(tp, dim=1)

    # filter low scored box
    # image_id class xywh
    if gt_boxes is not None:
        gt_boxes[:, 2:] *= img_size

    max_conf = torch.max(pred_anchor_boxes[0, :, 4])
    assert max_conf > 0
    if max_conf < score_thres:
        score_thres = torch.mean(pred_anchor_boxes[0, :, 4])

    return non_max_suppression_and_remove_overlap_with_gt(pred_anchor_boxes, gt_boxes, conf_thres=score_thres,
                                                          nms_thres=nms_thres, min_area=min_area)
