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

def return_dict_EDL(model, batch_x, batch_y, List):
    S ={}
    S[model.Deep['FL_layer0']] = batch_x
    S[model.classifier['Target'] ] = batch_y
    for i, element in enumerate(List):
        S[model.Trainer["random_matrices"][i]] = element
    return S

def sample_Z(X, m, n, kappa):
    return (X + np.random.uniform(-kappa, kappa, size=[m, n]))
    #return (X + np.random.normal(0, kappa, size=[m, n]))

#####################################################################################
def Analyse_custom_Optimizer_GDR(X_train, y_train, X_test, y_test, kappa):
    import gc
    global  Layer_Norm
    # Lets start with creating a model and then train batch wise.
    model = NN_class.learners()
    list = [inputs]
    list.extend([100 for i in xrange(10)])
    model = model.init_NN_custom(classes, 0.001, list, tf.nn.relu,'GDR')
    acc_array  = np.zeros( ( (Train_Glob_Iterations) , 1))
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
                List_2 = [0.05*g for g in grads_2[0]]
                List   = [np.add(a,b) for a,b in zip(List_1, List_2)]

                # temp = 0;
                # for j in range(0, 60, 2):
                #     Layer_Norm[i,temp] = np.linalg.norm(List[j])
                #     temp = temp + 1;

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

#####################################################################################
def Analyse_custom_Optimizer_EDL(X_train, y_train, X_test, y_test, kappa):
    import gc
    # Lets start with creating a model and then train batch wise.
    model = NN_class.learners()
    list = [inputs]
    list.extend([30 for i in xrange(50)])
    model = model.init_NN_custom(classes, 0.00001, list, tf.nn.tanh,'EDL')
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
                grads_1 = model.sess.run([ model.Trainer["matrix_output"] ],
                feed_dict ={ model.Deep['FL_layer0'] : batch_xs, model.classifier['Target']: batch_ys})
                List_2 = [g for g in grads_1[0]]
                # Gather Gradients
                _ = model.sess.run( [model.Trainer["EDL"]] , feed_dict= return_dict_EDL(model, batch_xs, batch_ys, List_2) )                 
                # _ = model.sess.run( [model.Trainer["EDL"]] , feed_dict= return_dict_EDL(model, batch_noise_xs, batch_ys, List_2) ) 

                temp = 0;
                for j in range(0, 50):
                    Layer_Norm[i,j] = np.linalg.norm(List_2[j])
                    temp = temp+1;

                # model.Summaries['train_writer'].add_summary(summary, i)

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



## Setup the parameters and call the functions
Temp =[]
Train_batch_size = 64
Train_Glob_Iterations = 100
# A global variable
Layer_Norm = np.zeros( ( (Train_Glob_Iterations) , 50))
classes = 10
print("I am working on EDL")
import tflearn
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels
print "Train, Test", X_train.shape, X_test.shape, y_train.shape, y_train.shape

inputs   = X_train.shape[1]
classes  = y_train.shape[1]
filename = 'vanishing_EDL_tanh_50.csv'
print("classes", classes)
print("filename", filename)
iterat_kappa = 1
Kappa_s = np.random.uniform(0, 1, size=[iterat_kappa])
print "kappa is", Kappa_s
for i in tqdm(xrange(iterat_kappa)):
    Temp.append(Analyse_custom_Optimizer_EDL(X_train,y_train,X_test,y_test, Kappa_s[i]))
print(np.array(Temp).mean(), np.array(Temp).std())
Results = np.zeros([iterat_kappa,2])
Results[:,1] = Temp
print "\n avg", Results[:,1].mean(), "std", Results[:,1].std()
Results[:,0] = Kappa_s[:]
np.savetxt(filename, Results, delimiter=',')

