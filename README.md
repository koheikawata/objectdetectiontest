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
