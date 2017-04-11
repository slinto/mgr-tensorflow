import os
import numpy as np
from six.moves import urllib
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

pwd = os.getcwd()

DECIMAL_LENGTH = 9
MODEL_PATH = pwd + '/data/output_graph.pb'
LABELS_PATH = pwd + '/data/output_labels.txt'

def create_graph():
    with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(image_url):
    # load image from url
    req = urllib.request.Request(image_url)
    response = urllib.request.urlopen(req)
    image_data = response.read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        # get top 5 predictions
        top_k = predictions.argsort()[-5:][::-1]
        f = open(LABELS_PATH, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]

        results = []

        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]

            results.append([human_string, score])
            print('%s (score = %.5f)' % (human_string, score))

        return results

def getNormalizedNumber(x):
    return round(float(x), DECIMAL_LENGTH)

def getNormalizedString(s):
    return s


# HTTP API
app = Flask(__name__)

@app.route('/api/photo-prediction', methods=['GET', 'POST'])
def photoPrediction():
    if request.method == 'POST':
        print(request.form['image_data'])
        answer = run_inference_on_image(request.form['image_data'])
    elif request.method == 'GET':
        answer = run_inference_on_image('http://www.woodenshoe.com/media/attila-graffiti-tulip.jpg')

    list = [
        {'id': getNormalizedString(answer[0][0]), 'value': getNormalizedNumber(answer[0][1])},
        {'id': getNormalizedString(answer[1][0]), 'value': getNormalizedNumber(answer[1][1])},
        {'id': getNormalizedString(answer[2][0]), 'value': getNormalizedNumber(answer[2][1])},
        {'id': getNormalizedString(answer[3][0]), 'value': getNormalizedNumber(answer[3][1])},
        {'id': getNormalizedString(answer[4][0]), 'value': getNormalizedNumber(answer[4][1])},
    ]
    return jsonify(status='OK', results=list)


@app.route('/api/photo-prediction-mock', methods=['GET', 'POST'])
def photoPredictionMock():
    if request.method == 'POST':
        print(request.form['image_data'])

    list = [
        {'id': 'prunus-serrulata', 'value': 0.99},
        {'id': 'salix-urbalix', 'value': 0.21},
        {'id': 'item-3', 'value': 0.03},
        {'id': 'item-4', 'value': 0.02},
        {'id': 'item-5', 'value': 0.01},
    ]
    return jsonify(status='OK', results=list)

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
