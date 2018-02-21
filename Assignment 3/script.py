import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


count = 0
def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))
    

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    #print(n_data)
    bias = np.ones((n_data,1))
    train_data = np.hstack((bias,train_data))
    initialWeights = np.reshape(initialWeights,(initialWeights.shape[0],1))
    z = np.dot(np.transpose(initialWeights), np.transpose(train_data))
    theta = sigmoid(z)
    #theta = np.log(theta)
    error = np.dot(np.log(theta),labeli)
    temp_ln = np.log(1-theta)
    temp_diff = 1-labeli
    error = error + np.dot(np.transpose(temp_diff),np.transpose(temp_ln))
    error = -1/n_data*error
    #print(error)
    
    
    
    
    error_grad = np.transpose(theta)-labeli
    error_grad = np.dot(np.transpose(train_data),error_grad)/n_data
    error_grad = error_grad.flatten()
    #print(error_grad)
    #sys.exit("Exit called")

    return error, error_grad



def blrPredict(W, data):
    """
    blrObjFunction predicts the label of data given the data and parameter W 
    of Logistic Regression
    
    Input:
        W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
        vector of a Logistic Regression classifier.
        X: the data matrix of size N x D
        
    Output: 
        label: vector of size N x 1 representing the predicted label of 
        corresponding feature vector given in data matrix
    
    """
    label = np.zeros((data.shape[0], 1))
    #print(W.shape)
    bias = np.ones((data.shape[0],1))
    data = np.hstack((bias,data))
    #print(data.shape)
    z = np.dot(np.transpose(W), np.transpose(data))
    z = np.transpose(z)
    #sig = sigmoid(z)
    #print(z.shape)
    #print(z[0])
    label = np.argmax(z,axis=1).reshape((data.shape[0],1))
    #print(label.shape)
    #print(sig.shape)
    
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    return label



def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    global count
    train_data, labels =args

    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    
    weights = params.reshape((n_feature+1,n_class))
    
    Data1 = np.ones((n_data,n_feature+1))
    Data1[:,1:] = train_data
    
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################

    
    temp = np.dot(Data1,weights)
    #print("exp values are:( 1st 10 values) ",  temp[1])
    nume = np.exp(temp)
    deme = np.sum(nume,axis=1)
    
    theta = np.divide(nume.transpose(),deme)   
    
    #print nume.shape, theta.shape, labels.shape
    theta = theta.transpose()
    
    #print( nume.shape, theta.shape, labels.shape)
    error = np.multiply(np.sum((np.multiply(np.log(theta),labels))), -1)/n_data
    
    error_grad = np.dot( (theta-labels).transpose(), Data1)/n_data
    
    #print(" 1st 10 values for error grad\n ", error_grad[:,1])
    count = count +1
    
    #print("\n error ", error, "count " ,  count)
    return error, error_grad.transpose().flatten()


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    #print label.shape

    ##################
    # YOUR CODE HERE #
    ##################
    
    Data1 = np.ones((data.shape[0],data.shape[1]+1))
    Data1[:,1:] = data
    #print W.shape ,    Data1.shape
    
    nume = np.exp(np.dot(Data1,W))
    deme = np.sum(nume,axis=1)
    
    #print deme.shape,nume.shape,nume[1],deme[1]
    label[:,0] = np.divide(nume.transpose(),deme).argmax(axis = 0)
    #print label.shape
    
    # HINT: Do not forget to add the bias term to your input data

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

clf = SVC(C = 100)
clf.fit(train_data, train_label)
svc_predict_train = clf.predict(train_data)
svc_predict_validation = clf.predict(validation_data)
svc_predict_test = clf.predict(test_data)
print('\n Training set Accuracy:' + str(accuracy_score(svc_predict_train, train_label)*100) + '%')
print('\n Validation set Accuracy:' + str(accuracy_score(svc_predict_validation, validation_label)*100) + '%')
print('\n Test set Accuracy:' + str(accuracy_score(svc_predict_test, test_label)*100) + '%')


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)

nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))
#print("final weights: 5  ",W_b[1],W_b[2],W_b[3],W_b[4],W_b[5])
count = 0

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
