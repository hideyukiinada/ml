This directory contains the code for a Flask-based webapp that allows the user to draw a digit on the web UI and let the NeuralNetwork class predict the digit.

![screenshot](https://github.com/hideyukiinada/ml/blob/master/assets/images/mnist-webapp-example.png)

*Steps
1. Open project/weight_persistence_mnist_example
2. Configure the below line to specify where to specify the weights file
```python
WEIGHTS_FILE_PATH = "../../weights/mnist_example.h5"
```
3. Run project/weight_persistence_mnist_example.
4. Verify that mnist_example.h5 has been generated. 
5. Go to ml/web_examples/mnist.
6. Start Python virtualenv if you haven't.  You need to be running Python 3.5 or later.
7. Type ./start.sh to start the Flask web server.
8. On your browser, go to http://localhost:5000

