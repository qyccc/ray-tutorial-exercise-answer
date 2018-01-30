from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import tensorflow as tf
import time

ray.init(num_cpus=4, redirect_output=True)

@ray.remote
class SimpleModel(object):
    def __init__(self):
        x_data = tf.placeholder(tf.float32, shape=[100])
        y_data = tf.placeholder(tf.float32, shape=[100])

        w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        b = tf.Variable(tf.zeros([1]))
        y = w * x_data + b

        self.loss = tf.reduce_mean(tf.square(y - y_data))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        grads = optimizer.compute_gradients(self.loss)
        self.train = optimizer.apply_gradients(grads)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()

        # Here we create the TensorFlowVariables object to assist with getting
        # and setting weights.
        self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)

        self.sess.run(init)

    def set_weights(self, weights):

        """Set the neural net weights.
        
        This method should assign the given weights to the neural net.
        
        Args:
            weights: Either a dict mapping strings (the variable names) to numpy
                arrays or a single flattened numpy array containing all of the
                concatenated weights.
        """
        # EXERCISE: You will want to use self.variables here.
        self.variables.set_flat(weights)
  
    def get_weights(self):
        """Get the neural net weights.
        
        This method should return the current neural net weights.
        
        Returns:
            Either a dict mapping strings (the variable names) to numpy arrays or
                a single flattened numpy array containing all of the concatenated
                weights.
        """
        # EXERCISE: You will want to use self.variables here.
        return self.variables.get_flat()
 
actors = [SimpleModel.remote() for _ in range(4)]

# Here 'weights' is a dictionary mapping variable names to the associated
# weights as a numpy array.
weights1=[]
#raise Exception('Implement this.')
for actor in actors:
    weight0 = actor.get_weights.remote()
    weights1.append(weight0)

#raise Exception('Implement this.')
weights1=ray.get(weights1)

print(weights1)
weight=np.mean(weights1)
p_arr=np.array([])
p_arr = np.concatenate((p_arr,[weight]))
p_arr = np.append(p_arr,weight) 

for actor in actors:
    actor.set_weights.remote(p_arr)


weights = ray.get([actor.get_weights.remote() for actor in actors])

for i in range(len(weights)):
    np.testing.assert_equal(weights[i], weights[0])

print('Success! The test passed.')