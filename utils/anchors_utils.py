import numpy as np
import tensorflow as tf
from glob import glob
import xmltodict

def make_center_anchors(anchors, grid_size=13):
    grid_range=tf.range(grid_size, dtype=tf.float32)
    xx, yy = tf.meshgrid(grid_range, grid_range)
    xy = tf.concat([tf.expand_dims(xx, -1), tf.expand_dims(yy, -1)],-1) + 0.5
    
    wh = anchors
    xy = tf.tile(tf.reshape(xy, (grid_size, grid_size, 1, 2)), (1,1,5,1))
    wh = tf.tile(tf.reshape(wh, (1,1,5,2)), (grid_size, grid_size, 1, 1))
    center_anchors = tf.concat([xy, wh], -1)

    return center_anchors

def load_objects(split):
    assert split in ["voc2007", "voc2012", "voc2007+voc2012", "custom"]
    split = split.split("+")
    data = []
    for sp in split:
        if(sp=="custom"):
            file_path = "C:/Users/HS/tensorflow_datasets/downloads/extracted/ZIP.data.zip/train_xmls/*.xml"
        else:
            file_path = "C:/Users/HS/tensorflow_datasets/downloads/extracted/TAR." + sp + "train.tar/VOCdevkit/" + sp.upper() + "/Annotations/*.xml"
        xml_list = glob(file_path)
        if(len(xml_list)!=0): print("loading " + sp)
        else: print("not exist data: " + sp)
        for xml in xml_list:
            f=open(xml)
            xml_file = xmltodict.parse(f.read())
            objs = xml_file['annotation']['object']
            objs = np.reshape(list([objs]),(-1,1))
            height = float(xml_file['annotation']['size']['height'])
            width = float(xml_file['annotation']['size']['width'])
            for obj in objs:
                xmax = float(obj[0]["bndbox"]["xmax"])
                xmin = float(obj[0]["bndbox"]["xmin"])
                ymax = float(obj[0]["bndbox"]["ymax"])
                ymin = float(obj[0]["bndbox"]["ymin"])
                w = (xmax - xmin)/width*13
                h = (ymax - ymin)/height*13
                data.append([w, h])
            f.close()
    return np.array(data)

def iou(boxes, clusters):
    box_w, box_h = boxes[..., 0:1], boxes[..., 1:2]
    clusters = np.transpose(clusters, (1,0))
    cluster_w, cluster_h = clusters[0:1], clusters[1:2]

    intersection = np.minimum(box_w, cluster_w) * np.minimum(box_h, cluster_h)
    union = box_w * box_h + cluster_w * cluster_h - intersection
    return intersection / union

def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(boxes, clusters), axis=1)])

def kmeans(boxes, k):
    num_boxes = boxes.shape[0]
    distances = np.empty((num_boxes, k))
    last_cluster = np.zeros((num_boxes,))

    np.random.seed()
    clusters = boxes[np.random.choice(num_boxes, k, replace=False)]

    while True:
        distances = 1 - iou(boxes, clusters)
        mean_distance = np.mean(distances)
        
        current_nearest = np.argmin(distances, axis=1)
        if(last_cluster == current_nearest).all():
            break
        for cluster in range(k):
            clusters[cluster] = np.mean(boxes[current_nearest == cluster], axis=0)

        last_cluster = current_nearest
    return clusters

def get_anchors(k, split):
    boxes = load_objects(split)
    result = kmeans(boxes, k)
    avg_acc = avg_iou(boxes, result)*100
    print("Average accuracy: {:.2f}%".format(avg_acc))
    result = np.array(sorted(np.array(result), key=lambda x: x[0]*x[1]), dtype=np.float32)
    print("Anchors")
    print(result)
    return result

#print(get_anchors(5,"voc2007+voc2012"))
# [[1.00737023, 1.68740245], 
# [2.50292653, 4.42315271], 
# [4.3078181, 8.78366732], 
# [7.83890834, 5.20955421], 
# [10.07927274, 10.72823056]]
#print(get_anchors(5,"custom"))
# [[1.4794683, 1.35702793], 
# [1.58223277, 2.75638217], 
# [2.88642237, 2.30851217], 
# [2.51967881, 4.26763735], 
# [4.82422153, 4.74087758]]