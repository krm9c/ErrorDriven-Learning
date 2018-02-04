
# Deep Learning Simulations
# Author : Krishnan Raghavan
# Date: Dec 25, 2016
#######################################################################################
# Define all the libraries
import os, sys, random, time, tflearn
import numpy as np
from   sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

####################################################################################
# Helper Function for the weight and the bias variable
# Weight
def xavier(fan_in, fan_out):
    low = -4*np.sqrt(4.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(4.0/(fan_in + fan_out))
    return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)

def weight_variable(shape, trainable, name):
  initial = xavier(shape[0], shape[1])
  return tf.Variable(initial, trainable = trainable, name = name)

# Bias function
def bias_variable(shape, trainable, name):
  initial = tf.random_normal(shape, trainable, stddev =1)
  return tf.Variable(initial, trainable = trainable, name = name)

#  Summaries for the variables
def variable_summaries(var, key):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'+key):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean'+key, mean)
        with tf.name_scope('stddev'+key):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev'+key, stddev)
        tf.summary.scalar('max'+key, tf.reduce_max(var))
        tf.summary.scalar('min'+key, tf.reduce_min(var))
        tf.summary.histogram('histogram'+key, var)

# Class
class learners():
    def __init__(self):
        self.classifier = {}
        self.Deep = {}
        self.Trainer = {}
        self.Evaluation = {}
        self.Summaries = {}
        self.keys = []
        self.sess = tf.InteractiveSession()

 ##############################
    # Function for defining every NN
    def nn_layer(self, input_tensor, input_dim, output_dim, act, trainability, key):
        with tf.name_scope(key):
            with tf.name_scope('weights'+key):
                self.classifier['Weight'+key] = weight_variable([input_dim, output_dim], trainable = trainability, name = 'Weight'+key)
                variable_summaries(self.classifier['Weight'+key], 'Weight'+key)
            with tf.name_scope('bias'+key):
                self.classifier['Bias'+key] = bias_variable([output_dim], trainable = trainability, name = 'Bias'+key)
                variable_summaries(self.classifier['Weight'+key], 'Weight'+key)
            with tf.name_scope('Wx_plus_b'+key):
                preactivate = tf.matmul(input_tensor, self.classifier['Weight'+key]) + self.classifier['Bias'+key]
                self.preactivations.append(preactivate)
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation'+key)
            tf.summary.histogram('activations', activations)
        return activations
    
    def Custom_Optimizer(self, lr):
        a_grad_net = tf.gradients(self.classifier['cost_NN'], self.keys)
        a = a_grad_net_1 + a_grad_net_2
        b = [  (tf.placeholder("float32", shape=grad.get_shape())) for grad in a]
        var_list_1 =[ item for item in self.keys_1]
        var_list_2 =[ item for item in self.keys_2]
        var_list = var_list_1+var_list_2
        c =  tf.train.AdamOptimizer(lr).apply_gradients( [ (e,var_list[i]) for i,e in enumerate(b) ] )
        return a, b, c

    def init_NN_custom(self, classes, lr, Layers, act_function):
        with tf.name_scope("FLearners_1"):
            self.Deep_1['FL_layer0'] = tf.placeholder(tf.float32, shape=[None, Layers[0]])
            for i in range(1,len(Layers)):
                self.Deep_1['FL_layer'+str(i)] = self.nn_layer(self.Deep_1['FL_layer'+str(i-1)], Layers[i-1],\
                Layers[i], act= act_function, trainability = False, key = 'FL_layer'+str(i))
                self.keys_1.append(self.classifier['Weight'+'FL_layer'+str(i)])
                self.keys_1.append(self.classifier['Bias'+'FL_layer'+str(i)])
                
        with tf.name_scope("Classifier"):
            self.classifier['class'] = self.nn_layer( self.Deep_1['FL_layer'+str(len(Layers)-1)],\
            Layers[len(Layers)-1], classes, act=tf.identity, trainability =  False, key = 'class')
            tf.summary.histogram('Output', self.classifier['class'])
            self.keys_1.append(self.classifier['Weightclass'])
            self.keys_1.append(self.classifier['Biasclass'])
        with tf.name_scope("Targets"):
            self.classifier['Target'] = tf.placeholder(tf.float32, shape=[None, classes])

        with tf.name_scope("Trainer"):
            Error_Loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = \
            self.classifier['class'], \
            labels = self.classifier['Target'], name='Cost'))

            # The final cost function 
            self.classifier["cost_NN"] = Error_Loss

            # Self writing optimizers
            self.Trainer["grads"], self.Trainer["grad_placeholder"], self.Trainer["apply_placeholder_op"] =\
             self.Custom_Optimizer(lr)
            tf.summary.scalar('LearningRate', lr)
            tf.summary.scalar('Cost_NN', self.classifier["cost_NN"])

            for grad in self.Trainer["grads"]:
                variable_summaries(grad, 'gradients')

            with tf.name_scope('Evaluation'):
                with tf.name_scope('CorrectPrediction'):
                    self.Evaluation['correct_prediction'] = tf.equal( tf.argmax(  tf.nn.softmax(self.classifier['class_1']
                    ) ,1), tf.argmax(self.classifier['Target'],1) )

                with tf.name_scope('Accuracy'):
                    self.Evaluation['accuracy'] = tf.reduce_mean(tf.cast(self.Evaluation['correct_prediction'], tf.float32))

                with tf.name_scope('Prob'):
                    self.Evaluation['prob'] = tf.cast( tf.nn.softmax(self.classifier['class_1']), tf.float32 )
                tf.summary.scalar('Accuracy', self.Evaluation['accuracy'])
                tf.summary.histogram('Prob', self.Evaluation['prob']) 
        
        self.Summaries['merged'] = tf.summary.merge_all()
        self.Summaries['train_writer'] = tf.summary.FileWriter('train/', self.sess.graph)
        self.Summaries['test_writer'] = tf.summary.FileWriter('test/')
        self.sess.run(tf.global_variables_initializer())
        return self
