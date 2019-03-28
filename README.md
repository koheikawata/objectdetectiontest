# objectdetectiontest

## Clone the repository to the local directory
```
$ git clone https://github.com/koheikawata/objectdetectiontest.git
```

## Dependency check
Dependency among Tensorflow, CUDA, cuDNN

https://www.tensorflow.org/install/source#tested_build_configurations

### Tensorflow
```
$ conda list | grep tensorflow
```

### CUDA version
```
$ nvcc --version
```
or
```
$ cat /usr/local/cuda/version.txt
```

### cuDNN version
```
$ cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
* Example
    * tensorflow_gpu-1.12.0
    * CUDA Version 9
    * cuDNN 7

### Install libraries
```
$ pip install -user cython
$ pip install -user contextlib2
$ pip install -user pillow
$ pip install -user lxml
$ pip install -user jupyter
$ pip install -user matplotlib
```

## Protbuf compilation
Compile the Protobuf libraries at research directory before using the framework. I saw erros many times, and then manually install and complie.
```
$ pwd
~/objectdetectiontest/research

$ wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
$ unzip protobuf.zip
$ ./bin/protoc object_detection/protos/*.proto --python_out=.
```

## Add libraries to PYTHONPATH
research and slim directory should be appended to PYTHONPATH. In research directory, run the command below.
```
$ pwd
~/objectdetectiontest/research

$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
## Test installation
To make sure if it is correctly installed, run the test command below in research directory
```
python object_detection/builders/model_builder_test.py
```
## Directory
```
.
├── research
|   ├── object_detection
|   |   ├── checkpoint_test1
|   |   |   └── eval_images
|   |   ├── legacy
|   |   |   ├── train.py
|   |   |   └── eval.py
|   |   └── output_test1
|   |       ├── Annotations
|   |       |   └── ***.xml
|   |       ├── JPEGImages
|   |       |   └── ***.jpg
|   |       ├── pascal_label_map.pbtxt
|   |       ├── train.record
|   |       └── val.record
|   ├── faster_rcnn_resnet101_coco_2018_01_28
|   |   └── model.ckpt
|   ├── faster_rcnn_resnet101_coco_test1.config
└───└── create_tf_record.py
```
## Create TFRecords
```
$ pwd
~/objectdetectiontest/research

$ python create_tf_record.py
```
## Download model
```
$ pwd
~/objectdetectiontest/research

$ wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
$ tar -xzvf faster_rcnn_resnet101_coco_2018_01_28.tar.gz
```
## Add path to faster_rcnn_resnet101_coco_test1.config
```
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "FULLPATH/research/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt"
  from_detection_checkpoint: true
}
train_input_reader {
  label_map_path: "FULLPATH/research/object_detection/output_test1/pascal_label_map.pbtxt"
  tf_record_input_reader {
    input_path: "FULLPATH/research/object_detection/output_test1/train.record"
  }
}
eval_config {
  max_evals: 100
  max_num_boxes_to_visualize: 50
  min_score_threshold: 0.3	
  num_examples: 160
  num_visualizations: 160
  skip_scores: false
  visualize_groundtruth_boxes: true
  keep_image_id_for_visualization_export: true
  include_metrics_per_category: true
  use_moving_averages: false
  visualization_export_dir: "FULLPATH/research/object_detection/checkpoint_test1/eval_images"
}
eval_input_reader {
  label_map_path: "FULLPATH/research/object_detection/output_test1/pascal_label_map.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "FULLPATH/research/object_detection/output_test1/val.record"
  }
}
```
## Train the model
```
$ python object_detection/legacy/train.py --logostderr --pipeline_config_path=faster_rcnn_resnet101_coco_test1.config --train_dir=checkpoint_test1
```
## Evaluate the model
```
$ python object_detection/legacy/eval.py --logostderr --pipeline_config_path=faster_rcnn_resnet101_coco_test1.config --checkpoint_dir=checkpoint_test1 --eval_dir=checkpoint_test1
```

## Launch Tensorboard
Add inbound port rule to Virtual Machine through Azure portal. In this case 6006 is open.
```
$ tensorboard --logdir=checkpoint_test1 --port 6006
```
Access through your browser
