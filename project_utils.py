import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble import GradientBoostingRegressor

def get_tree_oob_score(tree,X_train,y_train):
    #gets the oob score for the given tree
    indicies = _generate_unsampled_indices(tree.random_state, X_train.shape[0])
    y_true = y_train[indicies]
    y_hat_tree = tree.predict(X_train[indicies])
    rmse = np.sqrt(np.mean((y_true - y_hat_tree)**2))
    return rmse

def get_score_with_x_trees(x,predictions, y_val):
    #gets the RMSE for the top x trees in the list 
    predictions = predictions[:x]
    y_hat = predictions.mean(axis = 0)
    rmse = np.sqrt(np.mean((y_val - y_hat)**2))
    return rmse

def get_scores_n_fold(X,y,n_folds,parameters):
    
    kf = KFold(n_folds,shuffle=True)

    forest = RandomForestRegressor(n_estimators= parameters['n_estimators'],
                                   max_depth= parameters['max_depth'],
                                   n_jobs= -1, 
                                   oob_score= True, 
                                   max_features = parameters['max_features'],
                                   bootstrap= True, 
                                   min_samples_leaf = parameters['min_samples_leaf'],
                                   min_samples_split = parameters['min_samples_split'])
    
    results = []
    for train_ind, test_ind in kf.split(X):
        X_train, X_test, y_train, y_test = X[train_ind], X[test_ind], y[train_ind], y[test_ind]
        
        forest.fit(X_train,y_train)
        
        y_test_hat = forest.predict(X_test)

        #rmse of the entire model
        full_model_rmse = np.sqrt( np.mean( ((y_test - y_test_hat)**2)  ) )

        trees = forest.estimators_
        n_trees = len(trees)
        #the oob scores used to evaluate the trees 
        oob_scores = [get_tree_oob_score(tree,X_train,y_train) for tree in trees]


        #trees ordered by their oob scores
        tree_inds = np.argsort(oob_scores)
        ordered_trees = [trees[i] for i in tree_inds]
        ordered_tree_preds = np.array([tree.predict(X_test) for tree in ordered_trees])    

        #trees randomly ordered
        random_tree_preds = np.array([tree.predict(X_test) for tree in trees])
        
        #this gets the score for the best x trees for x from 1 to 1000   
        scores = []
        for x in range(1,n_trees+1):
            scores.append(get_score_with_x_trees(x,ordered_tree_preds,y_test))
            
        #this gets the score for a random x trees from 1 to 1000 as a null hypothesis
        random_scores = []
        for x in range(1,n_trees+1):
            random_scores.append(get_score_with_x_trees(x,random_tree_preds,y_test))
            
        results.append((full_model_rmse,scores,random_scores))
    
    return results

def run_experiment(X,y,num_trials,n_folds,parameters):
    results = []
    t1 = time.time()
    for x in range(0,num_trials):
        
        results.extend(get_scores_n_fold(X,y,10, parameters))
        
        if (x == 0):
            print("{}/{} done in {} seconds".format((x+1) * n_folds, 
                                                    num_trials * n_folds, 
                                                    round(time.time() -t1,3) ))
    print("{} models done in {} minutes".format(num_trials * n_folds, round((time.time() -t1)/60,3) ))

    return results

def train_gradient_boosters(n,X,y,n_estimators = 1000):
    #trains a bunch of gradient boosting classifiers 
    errors = []
    for i in range(0,n):
        gbr = GradientBoostingRegressor(n_estimators=n_estimators, 
                                        learning_rate=.001, 
                                        max_depth=3,subsample=.5, 
                                        max_features = None)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        gbr.fit(X_train,y_train)
        y_hat = gbr.predict(X_test)
        rmse = np.sqrt( np.mean( (y_test - y_hat)**2))
        errors.append(rmse)
    return errors