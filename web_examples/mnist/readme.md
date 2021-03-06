This directory contains the code for a Flask-based webapp that allows the user to draw a digit on the web UI and let the NeuralNetwork class predict the digit.

![screenshot](https://github.com/hideyukiinada/ml/blob/master/assets/images/mnist-webapp-example.png)

# Steps to set up and run the app
1. Open *project/weight_persistence_mnist_example*.
1. Configure the below line to specify where to save the weights file

    ```python
    WEIGHTS_FILE_PATH = "../../weights/mnist_example.h5"
    ```
1. Run *project/weight_persistence_mnist_example*.
1. Verify that you got 95% accuracy.  If this number is below 90%, something went wrong in training.
1. Verify that *mnist_example.h5* has been generated at the location specified in step 2. 
1. Go to *ml/web_examples/mnist/application* directory.
1. Open \_\_init__.py
1. Configure the below line to specify where to load the weights file from
 
    ```python
    WEIGHTS_FILE_PATH = "../../../weights/mnist_example.h5"
    ```
    Note that because the web app script is at a different level on the file system from the *weight_persistence_mnist_example* script, you need to adjust the directory level accordingly.  Namely, you cannot simply copy the relative path from step 2.
1. Start Python virtualenv if you haven't.  You need to be running Python 3.5 or later.
1. Type *./start.sh* to start the Flask web server.
1. On your browser, go to http://localhost:5000

# Accuracy
You'll notice that accuracy will be much lower than 95% when you draw a digit.  This is as expected because the model is not trained with your handwriting.  For example, I myself have difficulty in getting the system to recognize 8, 9 and 0 that I draw.

