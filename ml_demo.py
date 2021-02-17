# Splunk Presentation Demonstration - Alexander Hughes

# Lets start by importing our required libraries!

# Google's open-source machine learning framework
# Provides numerical computation, model building, and training algorithms
import tensorflow as tf

# Provides a neat Python interface for machine learning; runs on top of TensorFlow
from tensorflow import keras

# Extensive mathematical Python library for matrix operations 
import numpy as np


# Lets work through the Machine Learning process:
# First, lets 'gather' our data!

# Arrays of floating point (decimal) numbers on a cartesian plane
# Relationship between x and y is unknown (Spoiler, it's y = 3x + 1)
x_array = np.array([1.0, 2.0, -2.0, 0.0, -1.0, 5.0], dtype = 'float')
y_array = np.array([5.0, 8.0, -4.0, 2.0, -1.0, 17.0], dtype = 'float')

# data representing y = ln(x):
# x_array = np.array([5.0, 20.0, 3.0, 1.0, -0.50, -0.75], dtype = 'float')
# y_array = np.array([1.609, 2.996, 1.099, 0.0, -0.6931, -0.2877], dtype = 'float')

# Since we have the same amount of x and y coordinates, our data is pseudo-random,
# and our data is accurate, we don't need to do any pre-processing

# Next, lets initialize our model:
# This model is a single layer neural network 
# The layer has a single neuron, indicated by units = 1
# The input shape is [1], because we feed in one value, x, into the model
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape=[1])])

# Compiling (or preparing) the model requires the loss and optimizer functions
# Model makes a guess, then evaluates the strength of the guess with the loss function
# Model then uses the optimizer function to make another guess
# 'sgd' is stochastic gradient descent: iterative method for optimization
# 'mean_squared_error' is the average of the squares of the difference between the 
# estimated and actual values 
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

# Now, lets train our model!
# Relate the x_array to the y_array:
# num_epochs represents the number of times the model will observe the entire training data-set
num_epochs = 50
model.fit(x_array, y_array, epochs = num_epochs)

# Finally, let's run a prediction! 
x = 10.0
y = model.predict([x])
print("Prediction for x = " + str(x) + ": y = " + str(y) + "")

# Lets try a more complex function and see what happens
