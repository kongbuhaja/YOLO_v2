import tensorflow as tf
import math
import numpy as np

def get_hyper_params(**kwargs):
    hyper_params=dict()
    # hyper_params["anchors"] = np.array([[1.00737023, 1.68740245], 
    #                                     [2.50292653, 4.42315271], 
    #                                     [4.3078181, 8.78366732], 
    #                                     [7.83890834, 5.20955421], 
    #                                     [10.07927274, 10.72823056]], dtype=np.float32)
    hyper_params["anchors"]=np.array([[1.4794683, 1.35702793], 
                                        [1.58223277, 2.75638217], 
                                        [2.88642237, 2.30851217], 
                                        [2.51967881, 4.26763735], 
                                        [4.82422153, 4.74087758]], dtype=np.float32)
    hyper_params["iou_threshold"] = 0.3
    hyper_params["score_threshold"] = 0.4
    hyper_params["conf_threshold"] = 0.1
    hyper_params["image_size"] = 416
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value
    return hyper_params

def scheduler(epoch):
    if epoch < 100:
        return 1e-6
    elif epoch < 200:
        return 1e-5
    else:
        return 1e-6

def get_step_size(total_items, batch_size):
    return math.ceil(total_items / batch_size)

def generator(dataset):
    while True:
        for image_data in dataset:
            img, gt_grid = image_data
            yield img, gt_grid