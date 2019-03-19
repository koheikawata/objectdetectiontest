# objectdetectiontest

1. Clone the repository to the local directory
2. Change directory into research
```
$ cd objectdetectiontest/research
```
3. Protbuf compilation
Compile the Protobuf libraries before using the framework. I saw erros many times, and then manually install and complie.
```
$ pwd
~/objectdetectiontest/research

$ wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
$ unzip protobuf.zip
$ ./bin/protoc object_detection/protos/*.proto --python_out=.
```
4. Add libraries to PYTHONPATH
research and slim directory should be appended to PYTHONPATH. In research directory, run the command below.
```
$ pwd
~/objectdetectiontest/research

$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
To make sure if it is correctly installed, run the test command below in research directory
```
python object_detection/builders/model_builder_test.py
```
