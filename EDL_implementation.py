
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import operator
import tensorflow as tf
from functools import reduce



# The test file 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import Class_gen_Error as  NN_class
import tensorflow as tf
import gzip, cPickle
import numpy as np
import traceback
from tensorflow.examples.tutorials.mnist import input_data

###################################################################################
def import_pickled_data(string):
    f = gzip.open('../data/'+string+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    return X_train, y_train, X_test, y_test

###################################################################################
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

###################################################################################
def return_dict(placeholder, List, model, batch_x, batch_y):
    S ={}
    for i, element in enumerate(List):
        S[placeholder[i]] = element
    S[model.Deep['FL_layer0']    ] = batch_x
    S[model.classifier['Target'] ] = batch_y
    return S

def sample_Z(X, m, n, kappa):
    return (X + np.random.uniform(-kappa, kappa, size=[m, n]))
    #return (X + np.random.normal(0, kappa, size=[m, n]))

#####################################################################################
def Analyse_custom_Optimizer(X_train, y_train, X_test, y_test, kappa):
    import gc
    # Lets start with creating a model and then train batch wise.
    model = NN_class.learners()
    model = model.init_NN_custom(classes, 0.01, [inputs,100], tf.nn.relu)
    acc_array = np.zeros( ( (Train_Glob_Iterations) , 1))
    try:
        count = 0        
        t = xrange(Train_Glob_Iterations)
        from tqdm import tqdm
        Noise_data = sample_Z(X_test, X_test.shape[0], X_test.shape[1], kappa = kappa)

        for i in tqdm(t):
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys  = batch   
                batch_noise_xs  = sample_Z(batch_xs, Train_batch_size, X_train.shape[1], 1)
                # Gather Gradients
                grads_1 = model.sess.run([ model.Trainer["grads"] ],
                feed_dict ={ model.Deep['FL_layer0'] : batch_xs, model.classifier['Target']: batch_ys })
                grads_2 = model.sess.run([ model.Trainer["grads"] ],
                feed_dict ={ model.Deep['FL_layer0'] : batch_noise_xs, model.classifier['Target']: batch_ys }) 
                List_1 = [g for g in grads_1[0]]
                List_2 = [0.9*g for g in grads_2[0]]
                List = [np.add(a,b) for a,b in zip(List_1, List_2)]
                # Apply gradients
                summary, _ = model.sess.run( [ model.Summaries['merged'], model.Trainer["apply_placeholder_op"] ], \
                feed_dict= return_dict( model.Trainer["grad_placeholder"], List, model, batch_xs, batch_ys) )     
                #model.Summaries['train_writer'].add_summary(summary, i)

            if i % 1 == 0:
                summary, acc_array[i]  = model.sess.run( [ model.Summaries['merged'], model.Evaluation['accuracy'] ],\
                feed_dict={ model.Deep['FL_layer0']: X_test, model.classifier['Target'] : y_test})
                print("The accuracy is", acc_array[i])
                # model.Summaries['test_writer'].add_summary(summary, i)
                if max(acc_array) > 0.99:
                    summary, pr  = model.sess.run( [ model.Summaries['merged'], model.Evaluation['prob'] ], \
                    feed_dict ={ model.Deep['FL_layer0'] : X_test,model.classifier['Target'] : y_test } )
                    break
                # model.Summaries['test_writer'].add_summary(summary, i)
                
    except Exception as e:
        print e
        print "I found an exception"
        traceback.print_exc()
        tf.reset_default_graph()
        del model
        gc.collect()
        return 0
    tf.reset_default_graph()
    del model
    gc.collect()
    print "Accuracy", acc_array[i]
    return acc_array[i]


def get_num_nodes_respect_batch(tensor_shape):
    shape_start_index = 1 if tensor_shape.as_list()[0] is None else 0
    return reduce(operator.mul, tensor_shape.as_list()[shape_start_index:], 1), shape_start_index
    
def random_matrix(shape):
    with tf.variable_scope("radom_matrix"):
        rand_t = tf.random_uniform(shape, -1, 1)
        return tf.Variable(rand_t, name="weights")

def flatten_respect_batch(tensor):
    """
    Flattens a tensor respecting the batch dimension.
    Returns the flattened tensor and its shape as a list. Eg (tensor, shape_list).
    """
    with tf.variable_scope("flatten_respect_batch"):
        shape = tensor.get_shape()
        num_nodes, shape_start_index = get_num_nodes_respect_batch(shape)

        # Check if the tensor is already flat!
        if len(shape) - shape_start_index == 1:
            return tensor, shape.as_list()

        # Flatten the tensor respecting the batch.
        if shape_start_index > 0:
            flat = tf.reshape(tensor, [-1, num_nodes])
        else:
            flat = tf.reshape(tensor, [num_nodes])

        return flat

def reshape_respect_batch(tensor, out_shape_no_batch_list):
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

def direct_feedback_alignement(optimizer, loss, output, activation_param_pairs):
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
        # Get flatten size of outputs
        out_shape = output.get_shape()
        out_num_nodes, shape_start_index = get_num_nodes_respect_batch(out_shape)
        out_non_batch_shape = out_shape.as_list()[shape_start_index:]

        # Get the loss gradients with respect to the outputs.
        loss_grad = tf.gradients(loss, output)
        virtual_gradient_param_pairs = []
        # Construct direct feedback for each layer
        for i, (layer_out, layer_weights) in enumerate(activation_param_pairs):
            with tf.variable_scope("virtual_feedback_{}".format(i)):
                if layer_out is output:
                    proj_out = output
                else:
                    # Flatten the layer (this is naiive with respect to convolutions.)
                    flat_layer, layer_shape = flatten_respect_batch(layer_out)
                    layer_num_nodes = layer_shape[-1]

                    # First make random matrices to virutally connect each layer with the output.
                    rand_projection = random_matrix([layer_num_nodes, out_num_nodes])
                    flat_proj_out = tf.matmul(flat_layer, rand_projection)

                    # Reshape back to output dimensions and then get the gradients.
                    proj_out  = reshape_respect_batch(flat_proj_out, out_non_batch_shape)
                    factor = 0.0011
                for weight in layer_weights:
                    print(loss_grad, proj_out)
                    virtual_gradient_param_pairs +=  [
                        ( (tf.gradients(proj_out, weight, grad_ys=loss_grad)[0]+ factor*weight), weight)]

        train_op = optimizer.apply_gradients(virtual_gradient_param_pairs)
        return train_op 

import numpy as np
def sample_Z(X, m, n, kappa):
    #return (X + np.random.uniform(-kappa, kappa, size=[m, n]))
    return (X +  0*np.random.normal(0, kappa, size=[m, n]))

import tensorflow as tf
# pull MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data', one_hot=True)
# construction phase
x     = tf.placeholder(tf.float32, shape=[None, 784]) # placeholder for input data (images)
x_hat = tf.placeholder(tf.float32, shape=[None, 784]) # placeholder for input data (images)
y     = tf.placeholder(tf.float32, shape=[None, 10]) # placeholder for label data

with tf.name_scope('fc_0'): # first fully connected layer
    W0 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
    b0 = tf.Variable(tf.truncated_normal([300], stddev=0.1))
    h0 = tf.nn.relu(tf.matmul(x, W0) + b0)
    tf.summary.histogram('layer0_weights', W0) 

with tf.name_scope('fc_1'): # first fully connected layer
    W1 = tf.Variable(tf.truncated_normal([300, 200], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
    h = tf.nn.relu(tf.matmul(h0, W1) + b1)
    tf.summary.histogram('layer1_weights', W1) 

with tf.name_scope('fc_2'): # second fully connected layer
    W2 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
    y_predict = tf.matmul(h, W2) + b2

# NN 2
with tf.name_scope('fc_10'): # first fully connected layer
    W10 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
    b10 = tf.Variable(tf.truncated_normal([300], stddev=0.1))
    h10 = tf.nn.relu(tf.matmul(x_hat, W10) + b10)
    tf.summary.histogram('layer10_weights', W10) 

with tf.name_scope('fc_11'): # first fully connected layer
    W11 = tf.Variable(tf.truncated_normal([300, 200], stddev=0.1))
    b11 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
    h11 = tf.nn.relu(tf.matmul(h10, W11) + b11)
    tf.summary.histogram('layer11_weights', W11) 

with tf.name_scope('fc_12'): # second fully connected layer
    W12 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
    b12 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
    y_predict_1 = tf.matmul(h11, W12) + b12

with tf.name_scope('eval'): 
    with tf.name_scope('loss'): # calculating loss for the neural network
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))
        tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('eval1'): 
    with tf.name_scope('loss1'): # calculating loss for the neural network
        cross_entropy_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict_1))
        tf.summary.scalar('loss1', cross_entropy)

Total_loss_1 = (cross_entropy)   - 0.001*(cross_entropy*cross_entropy_1)
Total_loss_2 = (cross_entropy_1) - 0.001*(cross_entropy*cross_entropy_1)

dfa = direct_feedback_alignement(
        tf.train.AdamOptimizer(1e-4),
        Total_loss_1, y_predict,
        [(h, [W1, b1]),
         (h0, [W0, b0]),
         (y_predict, [W2, b2])])
         
dfa_1 = direct_feedback_alignement(
        tf.train.AdamOptimizer(1e-4),
        Total_loss_2, y_predict_1,
        [(h11, [W11, b11]),
         (h10, [W10, b10]),
         (y_predict_1, [W12, b12])])


correct = tf.equal(tf.argmax(y, 1), tf.argmax(0.5*(y_predict+y_predict_1), 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# execution phase
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # variable initialization step
train_steps = 20000
batch_size = 50

for i in range(train_steps):
    batch_x, batch_y = data.train.next_batch(batch_size) # collect next batch of input data and labels
    batch_x_noise = sample_Z(batch_x, batch_size, 784, 0.8)
    if i % 10 == 0:
        print("iterations", i, sess.run(accuracy, feed_dict={x: data.test.images, x_hat: data.test.images, y: data.test.labels}))
    else:
        sess.run(dfa,   feed_dict={x: batch_x, x_hat: batch_x_noise ,y: batch_y})
        sess.run(dfa_1, feed_dict={x: batch_x, x_hat: batch_x_noise ,y: batch_y})

# testing accuracy of trained neural network
print(sess.run(accuracy, feed_dict={x: data.test.images, x_hat: data.test.images, y: data.test.labels}))


