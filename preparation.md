# Data preparation guidelines

This document introduces the necessary setups before you can run the QBox algorithm. Please ensure that all the following procedures are performed correctly.

## 1. Download the pre-trained weight

We use the official pretrained weights, which can be downloaded by the script weights/download_yolov3_weights.sh. After that, please update the path of the weight file in the [config.py](config.py).

## 2. Create cfg file for the model

By default, each YOLO layer has 255 outputs: 85 values per anchor [4 box coordinates + 1 object confidence + 80 class confidences], times 3 anchors. Update the settings to filters=[5 + n] * 3 and classes=n, where n is your class count. This modification should be made in all 3 YOLO layers. (cf. the 5th procedure in https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data for more details.) Note that if you are using a new dataset, you should create a new .cfg file with the name "yolov3-spp-{target dataset name}.cfg" and put the file in the cfg folder.

## 3. Prepare for the raw data, and convert the annotation files into the specific format

Since we use the YOLOv3 implemented by https://github.com/ultralytics/yolov3/, please follow the instructions in this repository to complete this step. (https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data).

One constraint here is that the **the path to the label files should be obtained by replacing the 'images' in paths of the data with 'labels'.** For example, if the image path is `/aaa/voc/images/456.jpg`, the corresponding label file should be `/aaa/voc/labels/456.txt`.

## 4. Split data

The data should be split into the train/test set for the target dataset, and labeled/unlabeled set for the source dataset.

Please save the **absolute** file paths in the specific set as the following form:
```
/aaa/voc/images/456.jpg
/aaa/voc/images/457.jpg
/aaa/voc/images/458.jpg
...
```
The label files should be obtained by replacing the 'images' in paths of the data with 'labels'.

The following files are required: 
- `{dataset}_tgt_ini.txt`  
record the paths to the labeled data in target domain

- `{dataset}_tgt_test.txt`
record the paths to the test data in target domain

- `{dataset}_src.txt` 
record the paths to all data in source domain

- `{dataset}_src_ini.txt`
record the paths to the initially labeled data in source domain

- `{dataset}_src_unlab.txt`
record the paths to the unlabeled data for querying in source domain

Put these files in the same folder and pass the dir to the [qbox_main.py](qbox_main.py) when running the program.

## 5. Create label mapping file

Since the label space of source domain is larger than that in target domain, we only label the source instances in target classes. To this end, we need to know the mapping of the same class index from source to the target domain.

For example, the target classes are:
```
car
person
```

the source classes are
```
train
bicycle
person
traffic light
car
```

The label mapping should be a python dict
```
`src2tgt = {2:1, 4:0}`
```

Finally, please save the dict with pickle module.

## 6. Create .data file

If you are using a new dataset, please create the {dataset}.data file and put it in the "data" folder. this file should include the following entries:

- classes: the number of target classes

- source_train: the path to the file list file which includes all source data

- target_train: the path to the file list file which includes labeled target data

- valid: the path to the file list file which includes test data

- name: the path to the file which includes the class names


Here is an example:
```
classes=5
source_train=/data/dataset/city_src.txt
target_train=/data/dataset/kitti_tgt_ini.txt
valid=/data/dataset/kitti_tgt_test.txt
names=data/kitti_tgt.names
```

## 7. Update the entries in [config.py](config.py).

Here is an example:
```bash
KITTI_CITY = {
    'src': 'city',  # name of the source dataset. Should be accord with the split file, cfg file, .data file, .names file, etc.
    'tgt': 'kitti', # name of the target dataset. Should be accord with the split file, cfg file, .data file, .names file, etc.
    'pkl': '/data/saved_model/init_da_yolo_kitti.pkl',   # initialization model path (to eliminate the randomness)
    'src_dir': '/data/cityscape/coco/images/',
    'tgt_dir': '/data/download/kitti/tgt_images/',
    'data': 'data/kitti_city.data', # path to the .data file
    'label_map': 'data/city2kitti.pkl', # label map pickle file
    'cfg': 'cfg/yolov3-spp-kitti.cfg',  # path to the model cfg file
    'nc': 5,    # number of target classes
    'budget': 40000,    # budget for one query
    'min_area': 37.15854492187502   # the 10th percentile of the box area in the initially labeled set
}
```

## 8. Train the init model

Train an init model for active querying. **Make sure the `--save-dir` parameter is the same with that when you invoke qbox_main.py.**

```bash
# make sure the save path of the init model is accord with the config
python train_ini_model.py --dataset kitti --data-split-dir /data/dataset/ --save-dir /data/saved_model/
```


