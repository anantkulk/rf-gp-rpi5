import numpy as np

#Kernel Function
def kernel(x,x_prime,l, sigma_f):
    diff = x - x_prime
    d = np.dot(diff, diff)
    result = (sigma_f**2)*np.exp(-0.5*(d/(l**2)))
    return result
    
import pandas as pd

d = pd.read_csv("features.csv", header=None)

#Get the features data from csv
X = d.iloc[:, 1:18].values

#Formation Energy Targets
y = d.iloc[:, 18].values.reshape(-1,1)


#Normalize data for GP
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1.0
X_norm = (X-mean)/std

#Split into 400 for training and 100 for test
X_train = X_norm[:20]
X_test = X_norm[20:]

#Normalize data for GP
meany = y.mean(axis=0)
stdy = y.std(axis=0)
stdy[stdy == 0] = 1.0
y_norm = (y-meany)/stdy

Y_train = y_norm[:20]
Y_test = y_norm[20:]

def k_matrix(X_train,l,f):
    #Normalize Data
    
    n = X_train.shape[0]

    matrix = np.zeros((n,n))

    #Loop over the features in every materials and compute the kenerl between each point(n*n covariance matrix)
    for i in range(n):
        for j in range(n):    
            matrix[i,j] = kernel(X_train[i],X_train[j],l=l, sigma_f = f)
    return matrix 
    
def gp_train(X_train, Y_train,l,f,sigma_n):
    #Noise for Numerical Stability
    n = X_train.shape[0]
    K_y = k_matrix(X_train,l,f) + sigma_n*np.eye(n)

    #Cholesky Decomp
    # Ky = LL^T
    L = np.linalg.cholesky(K_y)

    #Lz=y
    z = np.linalg.solve(L,Y_train)

    #L^T(alpha) = z
    alpha = np.linalg.solve(L.T, z)
    return alpha, L



def gp_fit(X_train, X_test, Y_train,alpha, L, l,f):
    results = []
    sigmas = []
    # Build Kstar vector for prediction
    for j in X_test:
        k_star = []
        for i in X_train:
            k = kernel(i,j,l=l, sigma_f = f)
            k_star.append(k)
        k_array = np.array(k_star).reshape(-1,1)

        #prediction
        mean = np.dot(k_array.T,alpha)
        mean = (mean*stdy) + meany
        results.append(mean)

        #Uncertainity
        v = np.linalg.solve(L, k_star)
        var = kernel(j,j,l=l, sigma_f=f) - float(np.dot(v.T, v))
        s = (var*(stdy**2))**0.5
        sigmas.append(s)
    return np.array(results), np.array(sigmas)


# pool
X_test_fixed = X_test[380:]
Y_test_fixed = (Y_test[380:] * stdy) + meany  

X_pool = X_test[:380]
Y_pool = Y_test[:380]  


#Hyperparameter optimization
l_trials = [0.5,1.0,2.0,5.0,10.0]
f_trials = [0.5,1.0,2.0,5.0]
n_trials = [1e-5,1e-4,1e-3,1e-2,1e-1,1]

best_mae = np.inf
best_vals = None
uncertainity = None

for i in l_trials:
    for j in f_trials:
        for k in n_trials:
            trained = gp_train(X_train, Y_train,i,j,k)
            alpha = trained[0]
            L = trained[1]
            results = gp_fit(X_train,X_test_fixed,Y_train,alpha,L,i,j)
            mae = np.mean(abs(results[0] - Y_test_fixed))
            if mae < best_mae:
                best_mae = mae
                best_vals = [i,j,k]
                uncertainity = np.mean(results[1])
                
print(best_mae)
print(best_vals)
print(uncertainity)


iterations = 38
end_results = []
end_uncertainies = []


for i in range(iterations):
    #Training GP
    trained = gp_train(X_train, Y_train,l=5.0,f=1.0,sigma_n = 0.01)
    alpha = trained[0]
    L = trained[1]
    
    #Predict on Test set
    results = gp_fit(X_train,X_test_fixed,Y_train,alpha,L,l=5.0,f=1.0)
    mae = np.mean(abs(results[0] - Y_test_fixed))
    end_results.append(mae)
    end_uncertainies.append(np.mean(results[1]))
    
    #Uncertainity Sampling
    pool_eval = gp_fit(X_train,X_pool,Y_train,alpha,L,l=5.0,f=1.0)
    
    indices = np.argsort(pool_eval[1].flatten())[-10:]

    X_new = X_pool[indices]

    Y_new = Y_pool[indices]

    X_train = np.vstack([X_train, X_new])

    Y_train = np.vstack([Y_train, Y_new])

    mask = np.ones(len(X_pool), dtype=bool)
    mask[indices] = False

    
    X_pool = X_pool[mask]
    Y_pool = Y_pool[mask]

    if i % 5 == 0:
        print(f" iter: {i} done")

print(f"MAE Array: {end_results}")
print(f"Uncertainity Array: {end_uncertainies}")

import matplotlib.pyplot as plt


training = np.arange(30,410, 10)

plt.scatter(training, end_results)
plt.ylim(0,1)
plt.xlabel("Training Data")
plt.ylabel("MAE")
plt.title("Gaussian Process MAE")
