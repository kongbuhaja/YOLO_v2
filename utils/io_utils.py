import os
import argparse
import tensorflow as tf
from datetime import datetime

def get_log_path(model_type, custom_postfix=""):
    return "logs/{}{}/{}".format(model_type, custom_postfix, datetime.now().strftime("%Y%m%d-%H%M%S"))

def get_model_path(model_type, custom=False):
    main_path = "trained"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    filename="YOLOv2_{}_".format(model_type)
    filename += "model_weight"
    if(custom):
        filename += "_custom_dataset"
    filename += ".h5"
    model_path = os.path.join(main_path, filename)
    return model_path

def handle_args():
    parser = argparse.ArgumentParser(description="YOLOv2: You Only Look Once")
    parser.add_argument("-handle-gpu", action="store_true", help="Tensorflow 2 GPU compatibility flag")
    parser.add_argument("--backbone", required=False,
                         default="Darknet19",
                         metavar="['Darknet19']",
                         help="which backbone used for the YOLOv2")
    parser.add_argument("--custom_dataset_dir",
                         default=False,
                         help="custom_dataset dir")
    args = parser.parse_args()
    return args

def is_valid_backbone(backbone):
    assert backbone in ["Darknet19"]

def handle_gpu_compatibility():
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)