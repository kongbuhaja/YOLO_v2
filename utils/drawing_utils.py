import tensorflow as tf
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from utils import bbox_utils
import time

def draw_grid_map(img, grid_map, stride):
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    for grid in grid_map:
        draw.rectangle((
            grid[0] + stride // 2 - 2,
            grid[1] + stride // 2 - 2,
            grid[2] + stride // 2 + 2,
            grid[3] + stride // 2+2), fill=(255, 255, 255, 0))
    plt.figure()
    plt.imshow(image)
    plt.show()

def draw_bboxes(imgs, bboxes):
    colors = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, bboxes, colors)
    plt.figure()
    for img_with_bb in imgs_with_bb:
        plt.imshow(img_with_bb)
        plt.show()

def draw_bboxes_with_labels(img, pred, gt, all_labels, img_size):
    colors = tf.random.uniform((len(all_labels), 4), maxval=256, dtype=tf.int32)
    image_org = tf.keras.preprocessing.image.array_to_img(img)
    image_pred = image_org.copy()
    draw_org = ImageDraw.Draw(image_org)
    draw_pred = ImageDraw.Draw(image_pred)
    gt.append(draw_org)
    pred.append(draw_pred)
    
    for bboxes, labels, scores, draw in (gt, pred):
        for index, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = tf.split(bbox*img_size, 4)
            x1, y1, x2, y2 = tf.maximum(x1, 0.0), tf.maximum(y1, 0.0), tf.maximum(x2, 0.0), tf.maximum(y2, 0.0)
            if x2 <= x1 or y2 <= y1:
                continue
            label_index = int(labels[index])
            color = tuple(colors[label_index].numpy())
            label_text = "{0} {1:0.3f}".format(all_labels[label_index], float(scores[index]))
            draw.text((x1+4, y1+2), label_text, fill=color)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    plt.figure(figsize=(20,10))
    for i, image in enumerate((image_org, image_pred)):
        plt.subplot(1,2,i+1)
        plt.imshow(image)
    plt.show()

def draw_predictions(dataset, pred, hyper_params, batch_size):
    labels = hyper_params["labels"]
    img_size = hyper_params['image_size']
    for batch_id, image_data in enumerate(dataset):
        imgs, gt_grid = image_data
        img_size = imgs.shape[1]
        start = batch_id * batch_size
        end = start + batch_size
        start_time = time.time()
        batch_bboxes, batch_scores, batch_labels = bbox_utils.get_bboxes(pred[start:end], hyper_params, prediction=True)
        print("transform_pred: {:0.4f}s".format(time.time()-start_time))
        start_time=time.time()
        batch_gt_bboxes, batch_gt_scores, batch_gt_labels = bbox_utils.get_bboxes(gt_grid, hyper_params, prediction=False)
        print("transform_gt: {:0.4f}s".format(time.time()-start_time))
        for i, img in enumerate(imgs):
            draw_bboxes_with_labels(img, [batch_bboxes[i], batch_labels[i], batch_scores[i]], 
                                         [batch_gt_bboxes[i], batch_gt_labels[i], batch_gt_scores[i]], labels, img_size)