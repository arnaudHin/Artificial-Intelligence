## Author : Arnaud HINCELIN
## Date : Jan 3 2022
## Goal : Understand importance of normalisation


from sklearn.datasets import make_blobs
from matplotlib.pyplot import scatter, show, figure, contourf, colorbar, subplot, legend, title
from numpy import linspace, arange, meshgrid, concatenate, ravel, exp, c_, log, sum, zeros

#from ...CATS_DOGS import Cats_Dogs_FROM_SCRATCH as cdfs


def creat_dataset():
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape( (y.shape[0], 1) )

    print("shape of X : ", X.shape) #(100, 2)
    return (X, y)


def show_features(X, y):
    #print 2 features ; X[:,0], X[:,1]
    figure()
    scatter(X[:,0], X[:,1], c=y, cmap='plasma')
    show()



def show_cost_funtion_grid(X, y):
    #limite
    lim = 10
    h = 100

    W1 = linspace(-lim, lim, h) #tab 100, with value from -10 to 10
    W2 = linspace(-lim, lim, h) #tab 100, with value from -10 to 10

    #use a meshgrid = all configuration of each parameters
    W11, W22 = meshgrid(W1, W2)

    #rabse into only one tab : all configuration possible for all parameters
    W_final = c_[ W11.ravel(),W22.ravel() ].T #(2, 10000)
    print(W_final.shape)

    Z = X.dot(W_final) #(100, 2)*(2, 10000) = (100, 10000)
    A = 1/(1+exp(-Z)) #(100, 10000)

    #We have 10000 configurations
    #So we need the cost of each configuration, so we need 10000 costs
    #np.sum(X) -> sum of all -> it is a scalar ! shape=(1,)
    #np.sum(X, axis=0) -> sum of all colums -> it is a vector ! shape=(1, colums)
    #np.sum(X, axis=1) -> sum of all lines -> it is a vector ! shape=(1, lines)
    L = (-1/(A.shape[0]) ) * sum( y*log(A+1e-15) + (1-y) * (log(1-A+1e-15)), axis=0 ) #(10000,)
    print(L.shape) #(10000,)

    #Plot a contourplot, so need to reshape the costs
    L_r = L.reshape( (100,100) )



    #create an outlet feature
    X[:,1] = X[:,1]*10
    Z = X.dot(W_final) #(100, 2)*(2, 10000) = (100, 10000)
    A = 1/(1+exp(-Z)) #(100, 10000)

    #We have 10000 configurations
    #So we need the cost of each configuration, so we need 10000 costs
    #np.sum(X) -> sum of all -> it is a scalar ! shape=(1,)
    #np.sum(X, axis=0) -> sum of all colums -> it is a vector ! shape=(1, colums)
    #np.sum(X, axis=1) -> sum of all lines -> it is a vector ! shape=(1, lines)
    L2 = (-1/(A.shape[0]) ) * sum( y*log(A+1e-15) + (1-y) * (log(1-A+1e-15)), axis=0 ) #(10000,)
    print(L2.shape) #(10000,)

    #Plot a contourplot, so need to reshape the costs
    L2_r = L2.reshape( (100,100) )

    figure()
    subplot(1,2,1)
    contourf(W11, W22, L_r, levels=20, cmap='plasma')
    title('no_outlet')
    subplot(1,2,2)
    contourf(W11, W22, L2_r, levels=20, cmap='plasma')
    title('with_outlet')
    colorbar()
    show()
    #that shows cost evolution in function of all values of weights




#################################################
###                 MAIN PROGRAM              ###
#################################################

(X, y) = creat_dataset()
show_features(X, y)

#show grid cost in function of parameters
show_cost_funtion_grid(X, y)








