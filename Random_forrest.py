import pandas as pd
import numpy as np

#Same Across both models
d = pd.read_csv("features.csv", header=None)

#Get the features data from csv
X = d.iloc[:, 1:18].values

#Formation Energy Targets
y = d.iloc[:, 18].values.astype(float).flatten()



#Setup and normalize data
mean = X.mean(axis=0)
std = X.std(axis=0)
X_norm = (X-mean)/std

X_train = X_norm[:20]
X_test = X_norm[20:]


meany = y.mean(axis=0)
std[std == 0] = 1.0
stdy = y.std(axis=0)
y_norm = (y-meany)/stdy

Y_train = y_norm[:20]
Y_test = y_norm[20:]

def building_trees(X,y, depth=0, max_depth=20):
    #If Reaches max_depth or too little data in set stop
    if depth >= max_depth or len(y) < 6:
        return np.mean(y)

    #Choose random features
    n = int((17**0.5))
    pn = np.random.choice(17,n,replace=False)

    best_gain = -1*np.inf
    best_feature = None
    best_thres = None

    #Pick split points
    for i in pn:
        threshs = np.percentile(X[:,i],[25,50,75])
        for j in threshs:
            left = y[X[:,i] < j]
            right = y[X[:,i] >= j]
            
           #Evaluate set of split points on how well they maximize gain
            if len(left) > 0 and len(right) > 0:
                gain = np.var(y)-(len(left)/len(y))*np.var(left)-(len(right)/len(y))*np.var(right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_thres = j
    

    if best_feature is None:
        return np.mean(y)
    
    
    left_mask = X[:, best_feature] < best_thres
    l = building_trees(X[left_mask], y[left_mask], depth+1, max_depth)
    r = building_trees(X[~left_mask], y[~left_mask], depth+1, max_depth)
    return best_feature, best_thres, l, r 

def predict_tree(t, x):
    #if not tuple then it is a leaf node
    if not isinstance(t, tuple):
        return t
    feature, threshold, left, right = t
    #If tuple then continuing travelling down tree 
    if x[feature]< threshold:
        return predict_tree(left, x)
    else:
        return predict_tree(right, x)

def train_rf(X, y, n_trees=50, max_depth=10):
    #Creates all 50 trees that will make up the model
    trees = []
    for i in range(n_trees):
        indices = np.random.choice(len(X), len(X), replace=True)
        tree = building_trees(X[indices], y[indices], max_depth=max_depth)
        trees.append(tree)
    return trees


def predict_rf(trees, X):
    #final predictions using all 50 trees
    preds = np.array([[predict_tree(tree, x) for x in X] for tree in trees])
    return np.mean(preds, axis=0), np.std(preds, axis=0)
                    
#Same Active Learning Framework across both models
iterations = 38
end_results = []
end_uncertainies = []

#Set up fixed test and trainging sets
X_test_fixed = X_test[380:]
Y_test_fixed = (Y_test[380:] * stdy) + meany  

X_pool = X_test[:380]
Y_pool = Y_test[:380]  

for i in range(iterations):
    #Train Model
    trees = train_rf(X_train, Y_train, n_trees=50)
    #Predict Results
    results = predict_rf(trees, X_test_fixed)
    predict = (results[0]* stdy) + meany

    #Evaluate
    mae = np.mean(np.abs(Y_test_fixed - predict))
    end_results.append(mae)
    end_uncertainies.append(np.mean(results[1]))
    
    #Uncertainity sampling
    uncertainties = predict_rf(trees, X_pool)[1]
    indices = np.argsort(uncertainties)[-10:]

    X_new = X_pool[indices]

    Y_new = Y_pool[indices].flatten()

    X_train = np.vstack([X_train, X_new])

    Y_train = np.hstack([Y_train, Y_new])

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
plt.title("Random Forrest MAE")

