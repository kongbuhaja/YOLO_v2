import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
from utils import bbox_utils

def get_dataset(name, split, data_dir="~/tensorflow_datasets"):
    assert split in ['voc2007_train', 'voc2007_test', 'voc2012_train', 'voc2012_test', 'train', 'test']
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info

def preprocessing(image_data, final_height, final_width, augmentation_fn=None, evaluate=False):
    img = image_data["image"]
    gt_grid = image_data['grid']
    gt_labels = tf.cast(image_data["objects"]["label"], tf.int32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    if augmentation_fn:
        img, gt_grid = augmentation_fn(img, gt_grid)
    return img, gt_grid

def get_total_item_size(info, split):
    assert split in ['voc2007_train', 'voc2007_test', 'voc2012_train', 'voc2012_test', 'train', 'test']
    if split == "train+validation":
        return info.splits["train"].num_examples + info.splits["validation"].num_examples
    return info.splits[split].num_examples

def get_labels(info):
    return info.features['objects']['label'].names

def get_data_types():
    return tf.float32, tf.float32

def get_data_shapes():
    return ([None, None, None], [None, None, None, None])

def get_padding_values():
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32))