import os
import numpy as np
from six.moves import urllib
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

NUMBER_ROUND = 9

pwd = os.getcwd()
#imagePath = pwd + '/data/flower_photos/daisy/21652746_cc379e0eea_m.jpg'
#imagePath = 'http://www.kvhealthcare.org/Assets/Images/Quality/daisy/daisy.jpg'
#imagePath = 'http://www.woodenshoe.com/media/attila-graffiti-tulip.jpg'
modelFullPath = pwd + '/data/output_graph.pb'
labelsFullPath = pwd + '/data/output_labels.txt'


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image_url):
    answer = None

    req = urllib.request.Request(image_url)
    response = urllib.request.urlopen(req)
    image_data = response.read()

    # if not tf.gfile.Exists(imagePath):
    #     tf.logging.fatal('File does not exist %s', imagePath)
    #     return answer

    # image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]

        results = []

        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]

            results.append([human_string, score])
            print('%s (score = %.5f)' % (human_string, score))

        answer = labels[top_k[0]]
        print('answer')
        print(answer)
        return results

def getNormalizedNumber(x):
    return round(float(x), NUMBER_ROUND)

def getNormalizedString(s):
    return s
    #return s[2:-3]

## WEB_APP
app = Flask(__name__)

@app.route('/api/photo-prediction', methods=['GET', 'POST'])
def photoPrediction():
    print(request.form['image_data'])
    if request.method == 'POST':
        answer = run_inference_on_image(request.form['image_data'])
    elif request.method == 'GET':
        answer = run_inference_on_image('http://www.woodenshoe.com/media/attila-graffiti-tulip.jpg')

    list = [
        {'name': getNormalizedString(answer[0][0]), 'value': getNormalizedNumber(answer[0][1])},
        {'name': getNormalizedString(answer[1][0]), 'value': getNormalizedNumber(answer[1][1])},
        {'name': getNormalizedString(answer[2][0]), 'value': getNormalizedNumber(answer[2][1])},
        {'name': getNormalizedString(answer[3][0]), 'value': getNormalizedNumber(answer[3][1])},
        {'name': getNormalizedString(answer[4][0]), 'value': getNormalizedNumber(answer[4][1])},
    ]
    return jsonify(status='OK', results=list)

@app.route('/api/photo-prediction-mock', methods=['GET', 'POST'])
def photoPredictionMock():
    if request.method == 'POST':
        print(request.form['image_data'])

    list = [
        {'name': 'prunus-serrulata', 'value': 0.99},
        {'name': 'salix-urbalix', 'value': 0.21},
        {'name': 'item-3', 'value': 0.03},
        {'name': 'item-4', 'value': 0.02},
        {'name': 'item-5', 'value': 0.01},
    ]
    return jsonify(status='OK', results=list)

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
