
# Deep Learning Simulations
# The class file
# Author : Krishnan Raghavan
# Date: Dec 25, 2016
#######################################################################################
# Define all the libraries
import os, sys, random, time
import numpy as np
from   sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import operator
from functools import reduce

####################################################################################
# Helper Function for the weight and the bias variable initializations
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

# The main Class
class learners():
    def __init__(self):
        self.classifier = {}
        self.Deep = {}
        self.Trainer = {}
        self.Evaluation = {}
        self.Summaries = {}
        self.keys = []
        self.preactivations = []
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

    def get_num_nodes_respect_batch(self,tensor_shape):
        shape_start_index = 1 if tensor_shape.as_list()[0] is None else 0
        return reduce(operator.mul, tensor_shape.as_list()[shape_start_index:], 1), shape_start_index

    def random_matrix(self,shape):
        with tf.variable_scope("radom_matrix"):
            rand_t = tf.random_uniform(shape, -0.0001,0.0001)
            return tf.Variable(rand_t, name="weights")

    def flatten_respect_batch(self,tensor):
        """
        Flattens a tensor respecting the batch dimension.
        Returns the flattened tensor and its shape as a list. Eg (tensor, shape_list).
        """
        with tf.variable_scope("flatten_respect_batch"):
            shape = tensor.get_shape()
            num_nodes, shape_start_index = self.get_num_nodes_respect_batch(shape)

            # Check if the tensor is already flat!2
            if len(shape) - shape_start_index == 1:
                return tensor, shape.as_list()

            # Flatten the tensor respecting the batch.
            if shape_start_index > 0:
                flat = tf.reshape(tensor, [-1, num_nodes])
            else:
                flat = tf.reshape(tensor, [num_nodes])

            return flat

    def grad_placeholder(self, layer_grad, loss_grad):
        with tf.variable_scope("radom_matrix_1"):
            layer_grad = layer_grad+tf.random_normal(tf.shape(layer_grad), mean= 0.0, stddev = 0.001)
            s, u, v      = tf.svd(layer_grad)
            diag_mat     = tf.sqrt( tf.abs(tf.linalg.diag(s)) );
            mod_diag_mat = \
                tf.add(diag_mat, tf.multiply(0.00001, tf.eye(tf.shape(diag_mat)[1])))
            temp_rand    = tf.matmul(mod_diag_mat, v, adjoint_b=True)
            rand_t       = tf.matmul(u,temp_rand)
            ######### Choice 1
            # B = rand_t
            ######### Choice 2
            B = layer_grad
            ########## Choice 3
            # B = tf.random_uniform( tf.shape(layer_grad), -0.01,0.01)
            rand_t  = tf.matmul(B, loss_grad, transpose_a= True )[0];
            return rand_t

    def Custom_Optimizer(self, lr):
        a = tf.gradients(self.classifier['cost_NN'], self.keys)
        b = [  (tf.placeholder("float32", shape=grad.get_shape())) for grad in a]
        var_list =[ item for item in self.keys]
        c =  tf.train.AdamOptimizer(lr).apply_gradients( [ (e,var_list[i]) for i,e in enumerate(b) ] )
        return a,b, c

    def reshape_respect_batch(self,tensor, out_shape_no_batch_list):
        """
        Reshapes a tensor respecting the batch dimension.
        Returns the reshaped tensor
        """
        with tf.variable_scope("reshape_respect_batch"):
            tensor_shape = tensor.get_shape()
            shape_start_index = 1 if tensor_shape.as_list()[0] is None else 0

            # Flatten the tensor respecting the shape.
            if shape_start_index > 0:
                shaped = tf.reshape(tensor, [-1] + out_shape_no_batch_list)
            else:
                shaped = tf.reshape(tensor, out_shape_no_batch_list)
            return shaped

    def df_getmatrices(self, optimizer, loss, output, activation_param_pairs):
        with tf.variable_scope("direct_feedback_alignment"):
            # Matrix gradient list
            rand_list = []
            loss_grad = tf.gradients(loss, output)
            virtual_gradient_param_pairs = []

            # Construct direct feedback for each layer
            for i, (layer_out, layer_weights) in enumerate(activation_param_pairs):
                with tf.variable_scope("virtual_feedback_{}".format(i)):
                    if layer_out is output:
                        proj_out = output
                    else:
                        layer_grad = tf.gradients(loss, layer_out)
                        rand_t = self.grad_placeholder(layer_grad, loss_grad)
                        rand_list.append(rand_t);
            return rand_list

    def direct_feedback_alignement(self, optimizer, loss, output, activation_param_pairs, dec):
        """
        Builds a series of gradient ops which constitute direct_feedback_alignment.
        Params:
            - OPTIMIZER: A tf.train.Optimizer to apply to the direct feedback. Eg. tf.train.AdamOptimizer(1e-4)
            - LOSS: A loss function of the OUTPUTs to optimize.
            - OUTPUT: An output tensor for whatever tensorflow graph we would like to optimize.
            - ACTIVATION_PARAM_PAIRS: A list of pairs of output activations for every "layer" and the associated weight variables.
        Returns: a training operation similar to OPTIMIZER.minimize(LOSS).
        """
        with tf.variable_scope("direct_feedback_alignment"):
            # Matrix gradient list
            Mat_list =[]
            # Get flatten size of outputs
            out_shape = output.get_shape()
            out_num_nodes, shape_start_index = self.get_num_nodes_respect_batch(out_shape)
            out_non_batch_shape = out_shape.as_list()[shape_start_index:]
            # Get the loss gradients with respect to the outputs.
            loss_grad = tf.gradients(loss, output)
            virtual_gradient_param_pairs = []
            flag = 0;
            # Construct direct feedback for each layer
            for i, (layer_out, layer_weights) in enumerate(activation_param_pairs):
                with tf.variable_scope("virtual_feedback_{}".format(i)):
                    if layer_out is output:
                        proj_out = output

                        flag = 1
                    else:

                        flat_layer, layer_shape = self.flatten_respect_batch(layer_out)
                        layer_num_nodes = layer_shape[-1]
                        layer_grad = tf.gradients(output, layer_out)
                        rand_t = self.grad_placeholder(layer_grad, loss_grad)
                        Mat_list.append(tf.placeholder("float32", shape=rand_t.get_shape()))
                        layer_out = layer_out + tf.random_normal(tf.shape(layer_out), mean= 0.0, stddev = 0.001)
                        s,_,_    = tf.svd(layer_out)
                        diag_mat = tf.sqrt( tf.abs(tf.linalg.diag(s))+0.001);
                        sum_diag = (tf.reduce_sum(diag_mat)+0.001)
                        div_diag_mat = tf.truediv(diag_mat, sum_diag, name=None)
                        flat_proj_out = tf.matmul(flat_layer, Mat_list[len(Mat_list)-1])

                        # Reshape back to output dimensions and then get the gradients.
                        proj_out  = self.reshape_respect_batch(flat_proj_out, out_non_batch_shape)
                        fac = tf.add( tf.subtract(tf.eye(tf.shape(diag_mat)[1]), div_diag_mat)[0], 0.001*tf.eye(tf.shape(diag_mat)[1]))
                    j = 0;
                    for weight in layer_weights:
                        if flag == 0:
                            if j is 1:
                                reg  = dec*tf.squeeze( tf.matmul( fac  ,tf.expand_dims(weight,1) ), axis = 1)
                            else:
                                reg  = dec*tf.transpose(tf.matmul(fac  ,weight, transpose_b= True))
                        else:
                            reg = dec*weight
                        j= j+1;

                        virtual_gradient_param_pairs +=  [
                           ( tf.add( tf.gradients( proj_out , weight, grad_ys=loss_grad)[0], reg) , weight)]

            # I defines my variables here
            train_op = optimizer.apply_gradients(virtual_gradient_param_pairs)
            # print("start the optimizer")
            return train_op,  Mat_list


    def init_NN_custom(self, classes, lr, Layers, act_function, par='GDR'):
        if par =='EDL':
            array = []
        with tf.name_scope("FLearners_1"):
            self.Deep['FL_layer0'] = tf.placeholder(tf.float32, shape=[None, Layers[0]])
            for i in range(1,len(Layers)):
                self.Deep['FL_layer'+str(i)] = self.nn_layer(self.Deep['FL_layer'+str(i-1)], Layers[i-1],\
                Layers[i], act= act_function, trainability = False, key = 'FL_layer'+str(i))
                self.keys.append(self.classifier['Weight'+'FL_layer'+str(i)])
                self.keys.append(self.classifier['Bias'+'FL_layer'+str(i)])
                if par == 'EDL':
                    array.append(\
                    (self.Deep['FL_layer'+str(i)],\
                    ([self.classifier['Weight'+'FL_layer'+str(i)],self.classifier['Bias'+'FL_layer'+str(i)] ]) ) )


        with tf.name_scope("Classifier"):
            self.classifier['class'] = self.nn_layer( self.Deep['FL_layer'+str(len(Layers)-1)],\
            Layers[len(Layers)-1], classes, act=tf.identity, trainability =  False, key = 'class')
            tf.summary.histogram('Output', self.classifier['class'])
            self.keys.append(self.classifier['Weightclass'])
            self.keys.append(self.classifier['Biasclass'])
            if par == 'EDL':
                    array.append(\
                    ( self.classifier['class'],\
                    (self.classifier['Weightclass'],self.classifier['Biasclass']) ) )

        with tf.name_scope("Targets"):
            self.classifier['Target'] = tf.placeholder(tf.float32, shape=[None, classes])

        if par  is not 'EDL':
            with tf.name_scope("Trainer"):
                global_step = tf.Variable(0, trainable=False)
                learning_rate = lr

                # tf.train.exponential_decay(lr, global_step,
                # 100000, 0.90, staircase=True)

                Error_Loss =  tf.nn.softmax_cross_entropy_with_logits(logits = \
                self.classifier['class'], labels = self.classifier['Target'], name='Cost')
                Reg = 0.001;
                for element in self.keys:
                    Reg = Reg+tf.nn.l2_loss(element)
                # The final cost function
                self.classifier["cost_NN"] = tf.reduce_mean(Error_Loss + 0.001*Reg)
                # Self writing optimizers
                self.Trainer["grads"], self.Trainer["grad_placeholder"], self.Trainer["apply_placeholder_op"] =\
                self.Custom_Optimizer(learning_rate)

                print("Sumamries")
                tf.summary.scalar('LearningRate', lr)
                tf.summary.scalar('Cost_NN', self.classifier["cost_NN"])
                # for grad in self.Trainer["grads"]:
                #     variable_summaries(grad, 'gradients')
        else:
            with tf.name_scope("Trainer"):
                global_step = tf.Variable(0, trainable=False)
                learning_rate =lr
                dec = tf.train.exponential_decay(lr, global_step,
                                            100000, 0.99, staircase=True)
                Error_Loss =  tf.nn.softmax_cross_entropy_with_logits(logits = \
                self.classifier['class'], labels = self.classifier['Target'], name='Cost')

                # The final cost function
                self.classifier["cost_NN"] = tf.reduce_mean(Error_Loss)

                self.Trainer["matrix_output"] = self.df_getmatrices( tf.train.AdamOptimizer(learning_rate),
                    Error_Loss, self.classifier['class'], array )

                self.Trainer["EDL"] , self.Trainer["random_matrices"]= self.direct_feedback_alignement(
                    tf.train.AdamOptimizer(learning_rate),
                    Error_Loss, self.classifier['class'], array, dec)

        print("Evaluation")
        with tf.name_scope('Evaluation'):
            with tf.name_scope('CorrectPrediction'):
                self.Evaluation['correct_prediction'] = tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class']) ,1),\
                tf.argmax(self.classifier['Target'],1))

            with tf.name_scope('Accuracy'):
                self.Evaluation['accuracy'] = tf.reduce_mean(tf.cast(self.Evaluation['correct_prediction'], tf.float32))

            with tf.name_scope('Prob'):
                self.Evaluation['prob'] = tf.cast( tf.nn.softmax(self.classifier['class']), tf.float32 )
                tf.summary.scalar('Accuracy', self.Evaluation['accuracy'])
                tf.summary.histogram('Prob', self.Evaluation['prob'])

        self.Summaries['merged'] = tf.summary.merge_all()
        # self.Summaries['train_writer'] = tf.summary.FileWriter('train/', self.sess.graph)
        # self.Summaries['test_writer'] = tf.summary.FileWriter('test/')
        # print("initializing variables")
        self.sess.run(tf.global_variables_initializer())
        # print("return stuff")
        return self
