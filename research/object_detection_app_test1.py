import numpy as np
import os
import tensorflow as tf
import time
import json

from flask import Flask, redirect, request, Response, flash
from werkzeug.utils import secure_filename
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

THRESHOLD = 0.6
PORT = 5000
UPLOAD_DIR = 'object_detection/test1/flask_pics/'
LABEL_PATH = 'object_detection/test1/pascal_label_map.pbtxt'
MODEL_PATH = 'object_detection/test1/output/frozen_inference_graph.pb'

app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR

def load_model(model_path, label_path):
    g = tf.Graph()
    with g.as_default():
        with tf.gfile.GFile(model_path, "rb") as f:
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
        sess = tf.Session(graph=g)
    
    category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)

    return {'session': sess, 'image_tensor': image_tensor, 'tensor_dict': tensor_dict, 'category_index': category_index}

def inference(model, image_path):
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    start_time = time.time()
    output_dict = model['session'].run(model['tensor_dict'], feed_dict={model['image_tensor']: image_np_expanded})
    end_time = time.time()
    print('The inference takes {:.4f} sec'.format(end_time - start_time))

    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8).tolist()
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0].tolist()
    output_dict['detection_scores'] = output_dict['detection_scores'][0].tolist()
    
    result = []
    for idx, score in enumerate(output_dict['detection_scores']):
        if score > THRESHOLD:
            result.append({
                'class': output_dict['detection_classes'][idx],
                'label': model['category_index'][output_dict['detection_classes'][idx]]['name'],
                'confidence': output_dict['detection_scores'][idx],
                'bounding_box': output_dict['detection_boxes'][idx]
            })

    return (json.dumps(result))


@app.route('/inference_app', methods=['GET', 'POST'])
def inference_app():
    if request.method == 'POST':
        if 'file' not in request.files:
            return Response(response='Missing file', status=400) 
        model = app.config['MODEL']
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_DIR'], filename)
        file.save(filepath)

        try:
            response = Response(response=inference(model, filepath), status=200, mimetype='application/json')
        except Exception as e:
            response = Response(response=str(e), status=501)
        
        os.remove(filepath)
    
        return response
    return

if __name__ == '__main__':
    app.config['MODEL'] = load_model(MODEL_PATH, LABEL_PATH)
    app.run(host='0.0.0.0', port=PORT, debug=False)

