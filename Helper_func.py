# lets start with some imports
import random
import os, sys
from math import *
import numpy as np
from random import randrange
import tensorflow as tf


#1 -- Helper function for data 
# Now lets setup the data
def import_pickled_data(string):
    import gzip, cPickle
    f = gzip.open('../data/'+string+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    return X_train, y_train, X_test, y_test

# To collect samples
def extract_samples(X_f, y_f, p):
    index_1= [i for i,v in enumerate(y_f) if v == p]
    N = X_f[index_1,:]
    return N

# Bootstrap sample from the data
def subsample(dataset, ratio=1.0):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
        print("dataset length", len(dataset))
        sample.append(dataset[index])
        print "sample goes", sample
	return sample

### 2 - Next the classification helper function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis


# Gather data and setup HDR 
def classification_comparison(dataset, params, N):
    o_dim = params[1]
    g_size = params[0]
    flag = params[2]
    X,y,T,yT = import_pickled_data(dataset)
    # print(dataset, "is", X.shape)
    # #    Lets see how the dimension reduction
    P = np.zeros((N,9))
    score = np.zeros((N,9))
    for i in tqdm(xrange(N)):
        # DR -- 1 
        # Transform the training set
        # # Lets see how the dimension reduction
        Level, X_red_train = dim_reduction(X, i_dim=X.shape[1], o_dim =o_dim, \
        g_size=g_size, flag=flag)
        X_red_test=dim_reduction_test(T, Level, i_dim=X.shape[1], o_dim=o_dim,\
        g_size=g_size)
        # print("Dimensions reduced", X_red_train.shape, X_red_test.shape)
        # compare the classifiers
        names, score[i,:],  P[i,:] = comparison(X_red_train, y, X_red_test, yT)
    print("names", names)
    mean = np.round(score.mean(axis = 0),5)
    std  = np.round(score.std(axis = 0),5)
    s = ' '
    for i, element in enumerate(mean):
        s = s + ",("+str(element)+','+ str(std[i])+')' 

    print("Accuracy", s)
    mean = np.round(P.mean(axis = 0),5)
    std  = np.round(P.std(axis = 0), 5)
    s = ' '
    for i, element in enumerate(mean):
        s = s + ",("+str(element)+','+ str(std[i])+')' 
    print("p-value", s)


def comparison(XTrain, yTrain, XTest, yTest):
    names = ["Nearest Neighbors", "Linear SVM", 
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "LDA"]

    classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(), 
    LinearDiscriminantAnalysis()
    ]
    s =[]
    score = []
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(XTrain, yTrain)
        score.append(clf.score(XTest, yTest))
        labels = clf.predict(XTest)
        i = int(max(labels)-1)
        p_value =0
        index = [p for p,v in enumerate(yTest) if v == i]
        index = [ int(x) for x in index ]
        yTest= [ int(x) for x in yTest ]
        L = [v for p,v in enumerate(labels) if p not in index]
        p_value = ( (list(L).count(i)) )/float(len(labels));
        s.append(p_value)
    return names, np.array(score).reshape(1,9), np.array(s).reshape(1,9)


### 2 - Next the helper function for comparing dimension reduction
def dim_reduction_comparison(dataset, n_comp):
    from sklearn.decomposition import FactorAnalysis, PCA, KernelPCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding
    from sklearn import preprocessing
    from tqdm import tqdm
    
    N, y_train, T, y_test = import_pickled_data(dataset)
    name_1 = ["FA", "KPCA"]

    dims = \
    [
    FactorAnalysis(n_components= n_comp, tol=0.01, \
     copy=True, max_iter=1000, noise_variance_init=None,\
     svd_method='randomized', iterated_power=3, random_state=0),

    KernelPCA(n_components= n_comp, kernel='linear', gamma=None, degree=3, \
     coef0=1, kernel_params=None, alpha=1.0, \
     fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None,\
     remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=1),
    ]

    # Transform the train data-set
    scaler = preprocessing.StandardScaler(with_mean = True,\
     with_std = True).fit(N)
    X_train = scaler.transform(N)
    X_test = scaler.transform(T)
    epoch = 1

    for n, clf in zip(name_1, dims):
        scores = np.zeros((epoch,9))
        p_value = np.zeros((epoch,9))
        print("DR is", n)
        for i in tqdm(xrange(epoch)):
            Train = clf.fit_transform(X_train)
            Test =  clf.transform(X_test)
            names, scores[i,:], p_value[i,:] = comparison(Train, y_train, Test, y_test)
        
        print("names", names)
        mean = np.round(scores.mean(axis = 0),5)
        std  = np.round(scores.std(axis = 0),5)
        s = ' '
        for i, element in enumerate(mean):
            s = s + ",("+str(element)+','+ str(std[i])+')' 
        print("Accuracy", s)
        mean = np.round(p_value.mean(axis = 0),5)
        std  = np.round(p_value.std(axis = 0), 5)
        s = ' '
        for i, element in enumerate(mean):
            s = s + ",("+str(element)+','+ str(std[i])+')' 
        print("p-value", s)



# Reduce dimensions using the new approach
# 4 -- Reduction of dimension
def reduced_dimension_data(XTr, TTe, params):
    from Library_NMFE import dim_reductionNDR, dim_reductionNDR_test
    o_dim  = params[1]
    g_size = params[0]
    alpha  = params[2]

    Level, X_red_train = dim_reductionNDR(XTr, i_dim=XTr.shape[1], o_dim =o_dim, \
    g_size=g_size, alpha = alpha)

    # NDR
    X_red_test=dim_reductionNDR_test(TTe, Level, i_dim=TTe.shape[1], o_dim=o_dim,\
    g_size=g_size)

    return X_red_train, X_red_test



## 5 - Now detect the dimension reduced data 
def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


def cross_entropy_scores(features, target, weights):
    ll = np.sum( target*log(scores) - (1-target)*np.log(1 - scores))
    return ll

def xavier(fan_in, fan_out):
    low = -4*np.sqrt(4.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(4.0/(fan_in + fan_out))
    return np.random.uniform(low, high, [fan_in, fan_out])

def weight_variable( in_, out):
    initial = xavier( in_, out)
    return initial

 # Bias function
def bias_variable(in_, out):
    return np.random.normal(size = [in_ , out])

# Iterate Minibatches
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

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

# my (correct) solution:
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x/div

# Define the cost function
def cost(y, t):
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

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

def return_dict(placeholder, List, model, batch_x, batch_y):
    S ={}
    for i, element in enumerate(List):
        S[placeholder[i]] = element
    S[model.Deep['FL_layer0']    ] = batch_x
    S[model.classifier['Target'] ] = batch_y
    return S


def Analytic_Regression(model, Xtr, ytr, Xte, yte, iterate):
    print("In regression")
   
    try:
        t = xrange(iterate)
        from tqdm import tqdm
        for i in tqdm(t):
            for batch in iterate_minibatches(Xtr, ytr, 256, shuffle=True):
                batch_xs, batch_ys  = batch
                # Gather Gradients
                grads = model.sess.run([ model.Trainer["grads"] ],
                feed_dict ={ model.Deep['FL_layer0'] : batch_xs, model.classifier['Target']: batch_ys })
                List = [g for g in grads[0]]
                # Apply gradients
                summary, _ = model.sess.run( [ model.Summaries['merged'], model.Trainer["apply_placeholder_op"] ], \
                feed_dict= return_dict( model.Trainer["grad_placeholder"], List, model, batch_xs, batch_ys) )
            if i % 10 == 0:
                summary, a  = model.sess.run( [model.Summaries['merged'], model.Evaluation['accuracy']], feed_dict={ model.Deep['FL_layer0'] : \
                Xte, model.classifier['Target'] : yte})
                print("i", i, "--", a)
    except Exception as e:
        print(e)


def classification(X, y, XT, yT, iterate, classes):
    import Library_NNTensorFlow as Network_class

    print(classes)
    # Convert the tagets from one dimensional to one hot arrays
    print("y", y)
    print("yT", yT)
    y = np.equal.outer(y, np.arange(classes)).astype(np.float)
    yT = np.equal.outer(yT, np.arange(classes)).astype(np.float)
    # Lets start with creating a model and then train batch wise.
    inputs = X.shape[1];
    model = Network_class.Agent()
    model = model.init_NN_custom(classes, 0.001, [inputs, 100], tf.nn.relu)
    Analytic_Regression(model, X, y, XT, yT, iterate)
    #model = model.logistic_regression(X, y, classes, inputs, iterate)
    return model

## Addd noise to the data 
def noise_data(X, T, n_dim):
    ## n_dims say how many dimensions of noise to add
    dimensions_random = np.random.randint(X.shape[1], size= n_dim)
    for i, element in enumerate(dimensions_random):
        factor = np.random.randint(10)
        noise = np.random.normal(loc=2.0, scale=10.0, size=X.shape[0])
        
        
        temp = (factor*X[:,element]+noise).reshape([-1,1])
        X = np.append(X, temp, 1)       
        noise = np.random.normal(loc=2.0, scale=10.0, size=T.shape[0])
        temp = (factor*T[:,element]+noise).reshape([-1,1])
        T = np.append(T, temp, 1)   
    return X, T

def Add_noise_image(X_train, X_test, noise_dim):
    dim_prev = X_train.shape[1]
    X_train = X_train.reshape(-1,(X_train.shape[1]*X_train.shape[1]))
    X_test  = X_test.reshape(-1,(X_test.shape[1]*X_test.shape[1]))
    X_train_temp, X_test_temp = noise_data(X_train, X_test, noise_dim)

    # Convert the train array into images for CNN
    temp = X_train_temp[:,0:dim_prev*dim_prev]
    temp1 = X_train_temp[:,dim_prev*dim_prev:X_train_temp.shape[1]]
    dim = int(sqrt(X_train.shape[1]+noise_dim))
    temp_reshaped = temp.reshape(-1,dim_prev,dim_prev)
    X = np.zeros( (temp_reshaped.shape[0], dim, dim) );
    X[0:(temp_reshaped.shape[0]), 0:dim_prev, 0:dim_prev ] = temp_reshaped
    X[:, 0:dim_prev, (dim-1) ]    = temp1[:, 0:(int(temp1.shape[1]/2))]
    X[:,(dim-1),0:dim]            = temp1[:, (int(temp1.shape[1]/2)):int(temp1.shape[1])]
    
    # Convert the test array into images for CNN
    temp = X_test_temp[:,0:(dim_prev*dim_prev)]
    temp1 = X_test_temp[:, (dim_prev*dim_prev):X_test_temp.shape[1]]
    dim = int(sqrt(X_test.shape[1]+noise_dim))
    temp_reshaped = temp.reshape(-1,dim_prev,dim_prev)
    T = np.zeros( (temp_reshaped.shape[0], dim, dim) );
    T[0:(temp_reshaped.shape[0]), 0:dim_prev, 0:dim_prev ] = temp_reshaped
    T[:, 0:dim_prev, (dim-1) ]    = temp1[:, 0:(int(temp1.shape[1]/2))]
    T[:,(dim-1),0:dim]            = temp1[:, (int(temp1.shape[1]/2)):int(temp1.shape[1])]

    return X, T
