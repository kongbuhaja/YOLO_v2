import tensorflow as tf
import numpy as np
from utils import bbox_utils
import time

def init_stats(labels):
    stats = {}
    for i, label in enumerate(labels):
        stats[i] = {
            "label": label,
            "total": 0,
            "tp": [],
            "fp": [],
            "scores": []
        }
    return stats

def update_stats(pred_bboxes, pred_labels, pred_scores, gt_boxes, gt_labels, stats, hyper_params):
    iou_threshold = hyper_params["iou_threshold"]
    iou_map = bbox_utils.generate_iou_map(pred_bboxes, gt_boxes)
    merged_iou_map = tf.reduce_max(iou_map, axis=-1)
    max_indices_each_pred = tf.argmax(iou_map, axis=-1, output_type=tf.int32)
    sorted_ids = tf.argsort(merged_iou_map, direction="DESCENDING")
    count_holder = tf.unique_with_counts(tf.reshape(gt_labels, (-1,)))
    gt_boxes = tf.reshape(gt_boxes, (-1, 4))
    pred_bboxes = tf.reshape(pred_bboxes, (-1, 4))
    for i, gt_label in enumerate(count_holder[0]):
        if gt_boxes[i][2] <= gt_boxes[i][0] or gt_boxes[i][3] <= gt_boxes[i][1]:
            continue
        gt_label = int(gt_label)
        stats[gt_label]["total"] +=(count_holder[2][i])

    for batch_id, _ in enumerate(merged_iou_map):
        true_labels = []
        for i , sorted_id in enumerate(sorted_ids[batch_id]):
            if pred_bboxes[i][2] <= pred_bboxes[i][0] or pred_bboxes[i][3] <= pred_bboxes[i][1]:
                continue
            pred_label = int(pred_labels[batch_id, sorted_id])
            iou = merged_iou_map[batch_id, sorted_id]
            gt_id = max_indices_each_pred[batch_id, sorted_id]
            gt_label = int(gt_labels[batch_id, gt_id])
            score = pred_scores[batch_id, sorted_id]
            stats[pred_label]["scores"].append(score)
            stats[pred_label]["tp"].append(0)
            stats[pred_label]["fp"].append(0)
            if iou >= iou_threshold and pred_label == gt_label and gt_id not in true_labels:
                stats[pred_label]["tp"][-1] = 1
                true_labels.append(gt_id)
            else:
                stats[pred_label]["fp"][-1] = 1
    return stats

def calculate_ap(recall, precision):
    ap = 0
    for r in np.arange(0, 1.1, 0.1):
        prec_rec = precision[recall >= r]
        if len(prec_rec) > 0:
            ap += np.amax(prec_rec)
    ap /= 11
    return ap

def calculate_mAP(stats):
    aps = []
    for label in stats:
        label_stats = stats[label]
        tp = np.array(label_stats["tp"])
        fp = np.array(label_stats["fp"])
        scores = np.array(label_stats["scores"])
        ids = np.argsort(-scores)
        total = label_stats["total"]
        accumulated_tp = np.cumsum(tp[ids])
        accumulated_fp = np.cumsum(fp[ids])
        recall = accumulated_tp / total
        precision = accumulated_tp / (accumulated_fp + accumulated_tp)
        ap = calculate_ap(recall, precision)
        stats[label]["recall"] = recall
        stats[label]["precision"] = precision
        stats[label]["AP"] = ap
        aps.append(ap)
    mAP = np.mean(aps)
    return stats, mAP

def evaluate_predictions(dataset, pred, hyper_params, batch_size):
    labels = hyper_params["labels"]
    stats = init_stats(labels)
    for batch_id, image_data in enumerate(dataset):
        _, gt_grid = image_data
        start = batch_id * batch_size
        end = start + batch_size
        start_time = time.time()
        batch_bboxes, batch_scores, batch_labels = bbox_utils.get_bboxes(pred[start:end], hyper_params, prediction=True)
        print("transform_pred: {:0.4f}s".format(time.time()-start_time))
        start_time=time.time()
        batch_gt_bboxes, _, batch_gt_labels = bbox_utils.get_bboxes(gt_grid, hyper_params, prediction=False)
        print("transform_gt: {:0.4f}s".format(time.time()-start_time))
        start_time = time.time()
        stats = update_stats(batch_bboxes, batch_labels, batch_scores, batch_gt_bboxes, batch_gt_labels, stats, hyper_params)
        print("batch {} eval time: {:0.4f}s".format(batch_id, time.time()-start_time))
    stats, mAP = calculate_mAP(stats)
    print("mAP : {}".format(float(mAP)))
    return stats
        
        