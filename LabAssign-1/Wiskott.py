""" Regularized logistic regression model
using stochastic gradient descent """

#%%
import numpy as np
import matplotlib.pyplot as plt # %matplotlib qt for plots in separate window
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

""" Subtask 1: Visualization of the data """

def convert(str):
    """ converts string labels to binary labels """
    return 0 if str == 'Iris-versicolor' else 1

# Loading data
data = np.loadtxt('lab_iris_data.csv', encoding='utf-8',
                  delimiter=",", converters={3: convert})

class0 = np.matrix([sample for sample in data if sample[3] == 0]) # Iris-versicolor class
class1 = np.matrix([sample for sample in data if sample[3] == 1]) # Iris-virginia class

# 3-d plot of the three features of both classes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.set_title('Iris Data-set', fontweight='bold', fontsize=15)
ax.scatter(class0[:,0], class0[:,1], class0[:,2], s=10, label="Class 0", 
           c='b', alpha=0.3, marker='^') # 3d plot of the data points
ax.scatter(class1[:,0], class1[:,1], class1[:,2], s=10, label="Class 1",
           c='r', alpha=0.3, marker='o') # 3d plot of the data points
ax.legend()
ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2'); ax.set_zlabel('Feature 3')
#plt.savefig('1.png', dpi=1200)

# %%
""" Subtask 2: regularized logistic regression model using stochastic
gradient descent """

# optimal SGD parameters (empirically derived) 
T = 500 # number of iterations
k = 20 # Batch size for stochastic gradient descent
eta_init = 2  # Initial learning rate
lam = -0.7 # regularization parameter 

# Feature matrix
X = np.column_stack((np.ones((len(data))), # extends the matrix by the offset terms
                         data[:,0], # feature 1
                         data[:,1], # feature 2
                         data[:,2])) # feature 3
y = data[:,3] # labels

''' logistic regression hypothesis. Probability that sample x 
belongs to a certain class '''
h_theta = lambda x, theta: 1 / (1 + np.exp(np.dot(-x,  theta)))

def gradient(theta, X, y, lam):
    """ Computes gradient of the logistic regression loss function """
    sum = 0
    for i in range(X.shape[0]):
        sum += (h_theta(X[i], theta) - y[i]) * X[i].T
    sum += lam * np.sign(theta)
    
    return sum

def gradient_fast(theta, X, y, lam):
    """ Computes gradient of the logistic regression loss function 
    via Matrix vector multiplication """
    
    grad = X.T @ (h_theta(X, theta) - y)
    grad += lam * np.sign(theta)
    
    return grad

def SGD(X, y, T, k, eta_init, lam):
    """ Computes the stochastic gradient descent using T number of iterations, 
    k Batch size, eta_init initial learning rate and lam regularization parameter. 
    X feature matrix and y the label vector. """
    
    theta = np.zeros((X.shape[1])) # initial weight vector
    for t in range(1,T+1):
        eta = eta_init / np.sqrt(t) # adaptive learning rate
        # array of k random numbers chosen without replacement
        ran = np.random.choice(range(0, X.shape[0]), k, replace=False)
        """ Computes the new weight vector by using the randomly chosen 
        batch with size k, from the test set instead of using the entire
        set for the update. """
        theta -= (eta * (1/k) * gradient_fast(theta, X[ran], y[ran], lam)) 

    return theta

theta_star = SGD(X, y, T, k, eta_init, lam) # weight vector minimizing the loss function

def prediction(X, y, theta):
    ''' Checking the classification accuracy '''
    predictions = h_theta(X, theta_star) # labels predicted by trained model
    correct = 0 # count of correct predictions
    for ytrue, ypred in zip(y, predictions):
        """ if y=1: h(x_i) greater or equal 0.5.
            if y=0: h(x_i) less than 0.5."""
        if (ypred < 0.5 and ytrue == 0) or (ypred >= 0.5 and ytrue == 1):
            correct += 1 # increased if classification by model was correct 
    return correct # number of correct labels

def loss(theta, X, y, lam):
    """ Logistic regression loss """
    lr = 0
    for i in range(X.shape[0]):
        h = h_theta(X[i], theta)
        lr += -y[i] * np.log(h) - (1 - y[i]) * np.log(1 - h)
    lr += lam * np.sum(np.abs(theta))
    return lr / X.shape[0] # average over all samples

# Accuracy of the model
print(prediction(X, y, theta_star), " / ", len(X), " test samples correctly classified")
# Computing the loss
print('The logistic regresson loss: ', np.round(loss(theta_star, X, y, lam), 5))

#%%
""" Plotting the individual effects of the different parameters on-by-one. 
The other respective parameters are kept constant. 
Losses and classification accuracies are displayed as a function of the
respectigve parameters. """

# Testing the optimal regularizer
losses = []; accuracies = []
for i in range(-5,6,1): # computing the losses and accuracies for each regularizer
    theta_star = SGD(X, y, T, k, eta_init, i)
    losses.append(loss(theta_star, X, y, i))
    accuracies.append(prediction(X,y,theta_star))

plt.subplot(421); plt.xlabel('regularizer lambda', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.plot([i for i in range(-5,6,1)], losses, label='Loss')
plt.plot([-5,6], [0,0], c='orange', linestyle='-.') # line where loss == 0
plt.subplot(422); plt.xlabel('regularizer lambda', fontweight='bold')
plt.ylabel('Corr. class.', fontweight='bold')
plt.plot([i for i in range(-5,6,1)], accuracies, label='Accuracy',c='g')

# Testing the optimal initial learning rate
losses = []; accuracies = []
for i in range(-5,11,1): # computing the losses and accuracies for each lambda
    theta_star = SGD(X, y, T, k, i, lam)
    losses.append(loss(theta_star, X, y, lam))
    accuracies.append(prediction(X,y,theta_star))

plt.subplot(423); plt.xlabel('eta_init', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.plot([i for i in range(-5,11,1)], losses)
plt.plot([-5,11], [0,0], c='orange', linestyle='-.') # line where loss == 0
plt.subplot(424); plt.xlabel('eta_init', fontweight='bold')
plt.ylabel('Corr. class.', fontweight='bold')
plt.plot([i for i in range(-5,11,1)], accuracies,c='g')

# Test the optimal batch size
losses = []; accuracies = []
for i in range(1,X.shape[0]+1,5):
    theta_star = SGD(X, y, T, i, eta_init, lam)
    losses.append(loss(theta_star, X, y, lam))
    accuracies.append(prediction(X,y,theta_star))
    
plt.subplot(425); plt.xlabel('batch size k', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.plot([i for i in range(1,X.shape[0]+1,5)], losses)
plt.plot([-1,len(X)+1], [0,0], c='orange', linestyle='-.') # line where loss == 0
plt.subplot(426); plt.xlabel('batch size k', fontweight='bold')
plt.ylabel('Corr. class.', fontweight='bold')
plt.plot([i for i in range(1,X.shape[0]+1,5)], accuracies,c='g')

# Testing the optimal number of iterations
losses = []; accuracies = []
for i in range(100,2100,100):
    theta_star = SGD(X, y, i, k, eta_init, lam)
    losses.append(loss(theta_star, X, y, lam))
    accuracies.append(prediction(X,y,theta_star))
    
plt.subplot(427); plt.xlabel('Number of iterations T', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.plot([i for i in range(100,2100,100)], losses)
plt.plot([-100,2100], [0,0], c='orange', linestyle='-.') # line where loss == 0
plt.subplot(428); plt.ylabel('Corr. class.', fontweight='bold')
plt.xlabel('Number of iterations T', fontweight='bold')
plt.plot([i for i in range(100,2100,100)], accuracies,c='g')

# %%
""" Subtask 3: Plotting the linear separating hyperplane """

# Setting up the meshgrid for plotting the hyperplane
x,y = np.linspace(5,8), np.linspace(3,7)
x_,y_ = np.meshgrid(x,y)
# using theta[0] + theta[1]*x + theta[2]*y + theta[3]*z = 0 (plane equation)
z_ = (-theta_star[0] - theta_star[1] * x_ - theta_star[2] * y_) / theta_star[3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.set_title('Iris Data-set', fontweight='bold')
ax.scatter(class0[:,0], class0[:,1], class0[:,2], marker='o', 
           alpha=0.5, s=5, label="Class 0", c='b') # 3d plot of class 0
ax.scatter(class1[:,0], class1[:,1], class1[:,2], marker='o', 
           alpha=0.5, s=5, label="Class 1",c='r') # 3d plot of class 1
ax.plot_surface(x_, y_, z_, label='decision boundary', alpha=0.2,
                rstride=1, cstride=1, cmap=cm.Greens, linewidth=0,
                antialiased=False) # Separating hyperplane
ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2'); ax.set_zlabel('Feature 3')
# %%