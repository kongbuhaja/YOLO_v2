import tensorflow as tf
from utils import anchors_utils

def non_max_suppression(pred_bboxes, pred_labels, **kwargs):
    return tf.image.combined_non_max_suppression(pred_bboxes,
                                                 pred_labels,
                                                 **kwargs)


def generate_iou_map(bboxes, gt_boxes, transpose_perm=[0, 2, 1]):
    # input
    #     if 3d
    #     bboxes=(n,b,4)
    #     gtboxes=(n,g,4)
    # output
    #     intersection_arr / union_area=(n,b,g)

    gt_rank = tf.rank(gt_boxes)
    # if 3d gt_rank=3
    gt_expand_axis = gt_rank - 2

    # (n,b,1)
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=-1)
    # (n,g,1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)

    # (n,b)
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    # (n,g)
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)
    
    # bbox_x1=(n,b,1) gt_x1=(n,g,1) x_top=(n,b,g)
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, transpose_perm))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, transpose_perm))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, transpose_perm))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, transpose_perm))

    # (n,b,g)
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    # (n,b,1) + (n,1,g) = (n,b,g)
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, gt_expand_axis) - intersection_area)

    return intersection_area / union_area

def get_ious_from_grid(gt, pred):
    gx1, gy1, gx2, gy2 = tf.split(gt, 4, -1)
    px1, py1, px2, py2 = tf.split(pred, 4, -1)
    x1, y1 = tf.maximum(gx1, px1), tf.maximum(gy1, py1)
    x2, y2 = tf.minimum(gx2, px2), tf.minimum(gy2, py2)
    intersection = tf.maximum((y2-y1) * (x2-x1), 0.0)
    union = tf.maximum((gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - intersection, 1e-6);
    ious = intersection/union
    return ious

def center_to_corner(cxcy):
    x1y1 = cxcy[..., :2] - cxcy[..., 2:]/2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:]/2
    return tf.concat([x1y1, x2y2], -1)

def corner_to_center(xy):
    cxcy = (xy[..., :2] + xy[..., 2:]) /2 
    wh = xy[..., 2:] - xy[..., :2]
    return tf.concat([cxcy, wh], -1)

def get_bboxes(grid, hyper_params, prediction):
    anchors = hyper_params['anchors']
    B = anchors.shape[0]
    out_size = grid.shape[1]
    iou_threshold = 1
    score_threshold = 0
    conf_threshold = 0
    total_labels = len(hyper_params['labels'])

    xy = grid[..., :2]
    wh = grid[..., 2:4]
    confs = grid[..., 4:5]
    scores = grid[..., 5:]

    if(prediction):
        iou_threshold = hyper_params['iou_threshold']
        score_threshold = hyper_params['score_threshold']
        conf_threshold = hyper_params['conf_threshold']
        anchors = anchors_utils.make_center_anchors(anchors, out_size)
        xy = anchors[..., :2] + tf.math.sigmoid(xy)
        wh = anchors[..., 2:] * tf.math.exp(wh)
        confs = tf.math.sigmoid(confs)
        scores = tf.math.softmax(scores, -1)
    
    bboxes = center_to_corner(tf.concat([xy, wh], -1) / 13)
    bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)
    
    confs = tf.reshape(confs, (-1, out_size*out_size*B,1))
    scores = tf.reshape(scores, (-1, out_size*out_size*B,total_labels))
    bboxes = tf.reshape(bboxes, (-1, out_size*out_size*B,1,4))
    mask = tf.where(tf.greater(confs, conf_threshold), scores, tf.zeros_like(scores))

    out_boxes, out_scores, out_labels, _ = tf.image.combined_non_max_suppression(bboxes, mask, max_output_size_per_class=20, max_total_size=400,
                                            iou_threshold=iou_threshold, score_threshold=score_threshold)
    
    return out_boxes, out_scores, out_labels