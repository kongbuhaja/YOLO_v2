import tensorflow as tf
import numpy as np
from utils import bbox_utils, anchors_utils

class CustomLoss(object):
    def __init__(self, anchors, batch_size, coord=5, noojb=0.5):
        self.anchors = tf.reshape(anchors, (5,2))
        self.coord = coord
        self.noobj = noojb
        self.batch_size = batch_size;

    def yolov2_loss_fn(self, gt, pred):
        out_size = pred.shape[1]
        anchors = anchors_utils.make_center_anchors(self.anchors, out_size)
        pred_xy = tf.sigmoid(pred[..., :2]) + anchors[..., :2]
        pred_wh = anchors[..., 2:4] * tf.exp(pred[..., 2:4])
        pred_conf = tf.sigmoid(pred[..., 4:5])
        pred_cls = tf.math.softmax(pred[..., 5:], -1)
        
        # resp_mask, gt_xy, gt_wh, gt_conf, gt_cls = self.make_grid(gt_boxes, gt_labels, pred_xy, pred_wh, out_size)
        gt_xy, gt_wh, resp_mask, gt_cls = gt[..., :2], gt[..., 2:4], gt[..., 4:5], gt[..., 5:]
        no_resp_mask = 1 - resp_mask

        loc_xy_loss = self.coord * tf.reduce_sum(resp_mask * (gt_xy - pred_xy)**2)
        # loc_wh_loss = self.coord * tf.reduce_sum(gt_conf * tf.clip_by_value((tf.sqrt(gt_wh) - tf.sqrt(pred_wh))**2,1e-4,10000))
        loc_wh_loss = self.coord * tf.reduce_sum(resp_mask * (tf.sqrt(gt_wh) - tf.sqrt(pred_wh))**2)
        loc_loss = loc_xy_loss + loc_wh_loss
        
        gt_box = bbox_utils.center_to_corner(tf.concat([gt_xy, gt_wh],-1))
        pred_box = bbox_utils.center_to_corner(tf.concat([pred_xy, pred_wh],-1))
        ious = bbox_utils.get_ious_from_grid(gt_box, pred_box)
        gt_conf = ious * resp_mask
        conf_obj_loss = tf.reduce_sum(resp_mask * (gt_conf - pred_conf)**2)
        conf_noobj_loss = self.noobj * tf.reduce_sum(no_resp_mask * (gt_conf - pred_conf)**2)
        # conf_obj_loss = tf.reduce_sum(gt_conf * (gt_conf - pred_conf)**2)
        # conf_noobj_loss = self.noobj * tf.reduce_sum((1 - gt_conf) * (gt_conf - pred_conf)**2)
        conf_loss = conf_obj_loss + conf_noobj_loss

        cls_loss = tf.reduce_sum(resp_mask * (gt_cls - pred_cls)**2)

        return (loc_loss + conf_loss + cls_loss) / self.batch_size
