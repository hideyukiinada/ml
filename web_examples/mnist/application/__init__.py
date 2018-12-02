"""
MNIST digit recognition web example application.

Note
----
Before you run this app, you have to have weights file created to train the model.
To do so, run examples/weight_persistence_mnist_example.

Credit
------
The 20x20 digit area mentioned in the below article was very helpful in increasing the accuracy of my app:
[OK] Ole Kröger, Tensorflow, MNIST and your own handwritten digits, https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4

Other references
----------------
1. Gaussian matrix value is from:
Kernel (image processing), https://en.wikipedia.org/wiki/Kernel_(image_processing)

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import sys

sys.path.append("../..")  # To include NeuralNetwork modules

import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image

from project.neuralnetwork import NeuralNetwork
from project.neuralnetwork import Model
from project.neuralnetwork import Layer
from project.activationfunction import ActivationFunction as af
from project.costfunction import CostFunction as cf
from project.optimizer import Optimizer as opt

WEIGHTS_FILE_PATH = "../../../weights/mnist_example.h5"
TMP_FILE_PATH = "../../../tmp/mnist_digits/tmp.jpg"
INPUT_AREA_SIZE = 20
IMAGE_SIZE = 28

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

app = Flask(__name__)


def save_input_digit(c):
    """
    Save digit drawn for debugging

    Parameters
    ----------
    c: numpy array
        Digit drawn and post-processed
    """
    img_pil = Image.fromarray(np.uint8(c * 255.0))
    image_path = Path(TMP_FILE_PATH)
    img_pil.save(image_path)


def setup_predictor():
    """An example to show how to use NeuralNetwork class."""

    model = Model(num_input=28 * 28)
    model.add(Layer(512, activation=af.RELU))
    model.add(Layer(128, activation=af.RELU))
    model.add(Layer(10, activation=af.SIGMOID))

    nn = NeuralNetwork(model, learning_rate=0.001, cost_function=cf.CROSS_ENTROPY, optimizer=opt.SGD)

    weight_path = Path(WEIGHTS_FILE_PATH)
    if weight_path.exists():
        log.info("Weight file detected. Loading.")
        nn.load(weight_path)
    else:
        nn = None

    return nn


@app.route('/')
def root():
    return render_template('mnist.html', dimensions=range(INPUT_AREA_SIZE))


@app.route('/api/predict')
def predict_handler():
    """
    Process the array of values sent from the user, convert to numpy array and feed to the NeuralNetwork.
    Pass the prediction to the page to render.

    """
    one_count = 0  # Number of cells marked as 1
    c = np.zeros((INPUT_AREA_SIZE, INPUT_AREA_SIZE))  # Background is black (0)
    for i in range(INPUT_AREA_SIZE):
        for j in range(INPUT_AREA_SIZE):
            status = request.args.get("h_%d_%d" % (i, j), '')
            if status == '1':  # If white
                c[i, j] = 1.0  # set to 1
                one_count += 1

    # Apply gaussian blur
    # Set up kernel
    row_pad_count_top = 1
    row_pad_count_bottom = 1
    col_pad_count_left = 1
    col_pad_count_right = 1

    # Zero-pad to 22x22
    c_tmp = np.lib.pad(c, ((row_pad_count_top, row_pad_count_bottom), (col_pad_count_left, col_pad_count_right)),
                       'constant', constant_values=((0, 0), (0, 0)))

    c_out = np.zeros((INPUT_AREA_SIZE, INPUT_AREA_SIZE))
    gaussian3x3 = 1 / 16.0 * np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])

    # Commenting out as 5x5 kernel makes a character too thick.
    # gaussian5x5 = 1/16.0* np.array ([
    #    [1.0, 4.0, 6.0, 4.0, 1.0],
    #    [4.0, 16.0, 24.0, 16.0, 4.0],
    #    [6.0, 24.0, 36.0, 24.0, 6.0],
    #    [4.0, 16.0, 24.0, 16.0, 4.0],
    #    [1.0, 4.0, 6.0, 4.0, 1.0]
    # ])

    # Convolve
    for i in range(INPUT_AREA_SIZE):
        for j in range(INPUT_AREA_SIZE):
            c_out[i, j] = (c_tmp[i:i + 3, j:j + 3] * gaussian3x3).sum()

    # Make the brightest pixels pure white (1.0)
    max_val = c_out.max()
    multiplier = 1.0 / max_val
    c_out *= multiplier
    np.place(c_out, c_out > 0.6, [1.0])  # Another adjustment for brightness

    # Pad to 28x28
    row_pad_count_top = 4
    row_pad_count_bottom = 4
    col_pad_count_left = 4
    col_pad_count_right = 4
    c_padded = np.lib.pad(c_out, ((row_pad_count_top, row_pad_count_bottom), (col_pad_count_left, col_pad_count_right)),
                          'constant', constant_values=((0, 0), (0, 0)))
    #save_input_digit(c_padded)  # Uncomment this line to save and view the image on your file system

    x_flat = c_padded.reshape((1, IMAGE_SIZE * IMAGE_SIZE))
    y_hat_oh = nn.predict(x_flat)  # One-hot vector
    y_hat_int = np.argmax(y_hat_oh, axis=1)  # to int from one-hot vector
    result = int(np.squeeze(y_hat_int))
    return jsonify(result)


nn = setup_predictor()
