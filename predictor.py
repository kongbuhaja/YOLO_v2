import tensorflow as tf
from utils import bbox_utils, data_utils, drawing_utils, io_utils, train_utils, eval_utils

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

hyper_params = train_utils.get_hyper_params()
batch_size = 64
evaluate = False
custom_dataset_dir = args.custom_dataset_dir
backbone = args.backbone

io_utils.is_valid_backbone(backbone)

if backbone == "Darknet19":
    from models.Darknet19 import get_model, init_model

if custom_dataset_dir:
    test_data, info = data_utils.get_dataset("custom_dataset", "test", custom_dataset_dir)
    total_items = data_utils.get_total_item_size(info, "test")
else:
    test_data, info = data_utils.get_dataset("voc_dataset", "voc2007_test", "C:/Users/HS/tensorflow_datasets")
    total_items = data_utils.get_total_item_size(info, "voc2007_test")

labels = data_utils.get_labels(info)
hyper_params["labels"] = labels
img_size = hyper_params["image_size"]

data_types = data_utils.get_data_types()
data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()

test_data = test_data.map(lambda x: data_utils.preprocessing(x, img_size, img_size))

test_data = test_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)

YOLOv2_model = get_model(len(labels))
YOLOv2_model_path = io_utils.get_model_path(backbone, bool(custom_dataset_dir))
print(YOLOv2_model_path)
YOLOv2_model.load_weights(YOLOv2_model_path)

step_size = train_utils.get_step_size(total_items, batch_size)
pred_grid = YOLOv2_model.predict(test_data, steps=step_size, verbose=1)

if evaluate:
    eval_utils.evaluate_predictions(test_data, pred_grid, hyper_params, batch_size)
else:
    drawing_utils.draw_predictions(test_data, pred_grid, hyper_params, batch_size)