import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from utils import data_utils, train_utils, io_utils
from yolo_loss import CustomLoss
import augmentation

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

hyper_params = train_utils.get_hyper_params()
anchors=hyper_params["anchors"]
batch_size = 1
epochs = 500
load_weights = True
with_voc_2012 = True
backbone = args.backbone
custom_dataset_dir = args.custom_dataset_dir

io_utils.is_valid_backbone(backbone)

if backbone == "Darknet19":
    from models.Darknet19 import get_model, init_model

if custom_dataset_dir:
    train_data, info = data_utils.get_dataset("custom_dataset", "train", custom_dataset_dir)
    val_data, _ = data_utils.get_dataset("custom_dataset", "test", custom_dataset_dir)
    train_total_items = data_utils.get_total_item_size(info, "train")
    val_total_items = data_utils.get_total_item_size(info, "test")
else:
    train_data, info = data_utils.get_dataset("voc_dataset", "voc2007_train", "C:/Users/HS/tensorflow_datasets")
    val_data, _ = data_utils.get_dataset("voc_dataset", "voc2007_test", "C:/Users/HS/tensorflow_datasets")
    train_total_items = data_utils.get_total_item_size(info, "voc2007_train")
    val_total_items = data_utils.get_total_item_size(info, "voc2007_test")
    if with_voc_2012:
        voc_2012_data, voc_2012_info = data_utils.get_dataset("custom_dataset", "voc2012_train", "C:/Users/HS/tensorflow_datasets")
        voc_2012_total_items = data_utils.get_total_item_size(info, "voc2012_train")
        train_total_items += voc_2012_total_items
        train_data = train_data.concatenate(voc_2012_data)

labels = data_utils.get_labels(info)
hyper_params["total_labels"] = len(labels)
img_size = hyper_params["image_size"]

train_data = train_data.map(lambda x: data_utils.preprocessing(x, img_size,img_size, augmentation_fn=augmentation.apply))
val_data = val_data.map(lambda x: data_utils.preprocessing(x, img_size,img_size))

data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()

train_data = train_data.shuffle(batch_size*16).padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
val_data = val_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)


train_feed = train_utils.generator(train_data)
val_feed = train_utils.generator(val_data)

YOLOv2_model = get_model(hyper_params["total_labels"])
yolo_custom_loss = CustomLoss(anchors, batch_size)
YOLOv2_model.compile(optimizer=Adam(learning_rate=1e-3),
                   loss=[yolo_custom_loss.yolov2_loss_fn])#, yolo_custom_loss.conf_loss_fn, yolo_custom_loss.cls_loss_fn])
init_model(YOLOv2_model)

YOLOv2_model_path = io_utils.get_model_path(backbone, bool(custom_dataset_dir))
print(YOLOv2_model_path)
if load_weights:
    YOLOv2_model.load_weights(YOLOv2_model_path)
YOLOv2_log_path = io_utils.get_log_path(backbone)

checkpoint_callback = ModelCheckpoint(YOLOv2_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=YOLOv2_log_path)
learning_rate_callback = LearningRateScheduler(train_utils.scheduler, verbose=0)

step_size_train = train_utils.get_step_size(train_total_items, batch_size)
step_size_val = train_utils.get_step_size(val_total_items, batch_size)

YOLOv2_model.fit(train_feed,
               steps_per_epoch=step_size_train,
               validation_data=val_feed,
               validation_steps=step_size_val,
               epochs=epochs,
               callbacks=[checkpoint_callback, tensorboard_callback, learning_rate_callback])