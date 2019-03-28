import hashlib
import io
import logging
import os
import random

import contextlib2
from lxml import etree
import PIL.Image
import tensorflow as tf
import pandas as pd

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')
flags.DEFINE_float('ratio_train', 0.8, 'Ratio of train and val')
flags.DEFINE_string('label_map_path', 'object_detection/output_test1/pascal_label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_string('image_dir', 'object_detection/output_test1/JPEGImages', 'Path to jpg image directory')
flags.DEFINE_string('annotation_dir', 'object_detection/output_test1/Annotations', 'Path to annotation xml directory')
flags.DEFINE_string('train_output_path', 'object_detection/output_test1/train.record', 'Path to train output TFRecords')
flags.DEFINE_string('val_output_path', 'object_detection/output_test1/val.record', 'Path to val output TFRecords')
FLAGS = flags.FLAGS

def dict_to_tf_example(data, label_map_dict, image_dir):
    img_path = os.path.join(image_dir, data['filename'] + '.jpg')
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    
    for obj in data['object']:
        xmin = float(obj['bndbox']['xmin'])
        xmax = float(obj['bndbox']['xmax'])
        ymin = float(obj['bndbox']['ymin'])
        ymax = float(obj['bndbox']['ymax'])
        
        xmins.append(xmin / width)
        ymins.append(ymin / height)
        xmaxs.append(xmax / width)
        ymaxs.append(ymax / height)
        
        class_name = obj['name']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])
    
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
    return tf_example

def create_tf_record(output_filename, num_shards, label_map_dict, image_dir, annotation_dir, examples):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        xml_path = os.path.join(annotation_dir, example + '.xml')
        with tf.gfile.GFile(xml_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()

def main(_):
    label_map_dict = label_map_util.get_label_map_dict(os.path.join(os.getcwd(), FLAGS.label_map_path))
    image_dir = os.path.join(os.getcwd(), FLAGS.image_dir)
    annotation_dir = os.path.join(os.getcwd(), FLAGS.annotation_dir)

    examples_list = []
    xml_filelist = os.listdir(annotation_dir)
    for i in xml_filelist:
        if(i[-4:] == '.xml'):
            xml_filename, ext = os.path.splitext(i)
            examples_list.append(xml_filename)
    
    random.seed()
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(FLAGS.ratio_train * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    train_output_path = os.path.join(os.getcwd(), FLAGS.train_output_path)
    val_output_path = os.path.join(os.getcwd(), FLAGS.val_output_path)

    create_tf_record(train_output_path, FLAGS.num_shards, label_map_dict, image_dir, annotation_dir, train_examples)
    create_tf_record(val_output_path, FLAGS.num_shards, label_map_dict, image_dir, annotation_dir, val_examples)

if __name__ == '__main__':
    tf.app.run()







