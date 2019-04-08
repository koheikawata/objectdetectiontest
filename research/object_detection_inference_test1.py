import numpy as np
import os
import tensorflow as tf
import time
import json

from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

THRESHOLD = 0.6
LABEL_PATH = 'object_detection/test1/pascal_label_map.pbtxt'
MODEL_PATH = 'object_detection/test1/output/frozen_inference_graph.pb'
IMAGE_PATH = 'object_detection/test1/JPEGImages/IMG_00000.jpg'

image = Image.open(IMAGE_PATH)
(im_width, im_height) = image.size
image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
image_np_expanded = np.expand_dims(image_np, axis=0)

with tf.gfile.GFile(MODEL_PATH, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

tf.import_graph_def(graph_def, name='')
ops = tf.get_default_graph().get_operations()
all_tensor_names = {output.name for op in ops for output in op.outputs}
tensor_dict = {}
for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

start_time = time.time()
with tf.Session() as sess:
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})
end_time = time.time()
print('Inference takes {:.4f} sec'.format(end_time - start_time))

output_dict['num_detections'] = int(output_dict['num_detections'][0])
output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8).tolist()
output_dict['detection_boxes'] = output_dict['detection_boxes'][0].tolist()
output_dict['detection_scores'] = output_dict['detection_scores'][0].tolist()

category_index = label_map_util.create_category_index_from_labelmap(LABEL_PATH, use_display_name=True)

result = []
for idx, score in enumerate(output_dict['detection_scores']):
    if score > THRESHOLD:
        result.append({
            'class': output_dict['detection_classes'][idx],
            'label': category_index[output_dict['detection_classes'][idx]]['name'],
            'confidence': output_dict['detection_scores'][idx],
            'bounding_box': output_dict['detection_boxes'][idx]
        })

json.dumps(result)