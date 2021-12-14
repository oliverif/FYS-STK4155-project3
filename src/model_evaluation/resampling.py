from sklearn.model_selection import KFold,train_test_split
from sklearn.utils import resample
from numpy import mean, empty, var
from .metrics import METRIC_FUNC

def bootstrap(model, X_train, X_test, z_train,n_bootstraps=200):

    z_pred = empty((X_test.shape[0], n_bootstraps))   
    for i in range(n_bootstraps):
        X_,z_ = resample(X_train, z_train)         #random resampling 
        model.fit(X_,z_)                #fit beta to new resampled data
        z_pred[:,i] = model.predict(X_test).ravel() #predict test data

    return z_pred

def bias_var(model, X_train,X_test, z_train,z_test, n_bootstraps=200):
    
    z_pred = bootstrap(model,X_train, X_test, z_train,n_bootstraps)
    error = mean( mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias = mean( (z_test - mean(z_pred, axis=1, keepdims=True))**2 )
    variance = mean( var(z_pred, axis=1, keepdims=True) )
    
    return error, bias, variance

def cross_validate(model, X, z, k_folds=5, X_scaler=None, z_scaler = None, metrics = None):
    '''
    Calculates the cross validated score of model.
    The scoring metric  is given by the model itself.
    Ex. 
    model is a classifier -> metric is accuracy.
    model is a regressor -> metric is R2.
    '''
    
    kfold = KFold(n_splits = k_folds, shuffle=True) 
    if (metrics is not None):
        train_scores = {}
        test_scores = {}
        for metric in metrics:
            train_scores[metric] = list()
            test_scores[metric] = list()
        
    else:
        train_scores = []
        test_scores = []
    
    for train_inds, test_inds in kfold.split(X):
        X_train = X[train_inds]
        z_train = z[train_inds]

        X_test = X[test_inds]
        z_test = z[test_inds]
        
        if(X_scaler is not None):
            X_train = X_scaler.fit_transform(X_train)
            X_test = X_scaler.transform(X_test)
 
        if(z_scaler is not None):
            z_train = z_scaler.fit_transform(z_train.reshape(-1,1))
            z_test = z_scaler.transform(z_test.reshape(-1,1)) 
        
        model.fit(X_train, z_train)
        if(metrics is not None):
            p_train = model.predict(X_train)
            p_test = model.predict(X_test)
            for metric in metrics:
                train_scores[metric].append(METRIC_FUNC[metric](z_train,p_train.reshape(-1,1)))
                test_scores[metric].append(METRIC_FUNC[metric](z_test,p_test.reshape(-1,1)))
                
        else:
            train_scores.append(model.score(X_train, z_train))
            test_scores.append(model.score(X_test, z_test))

    return {'train_scores':train_scores, 'test_scores':test_scores}

def cross_val_score(model, X, z, k_folds, X_scaler=None, z_scaler = None, metrics = None):
    '''
    Runs cross_validate and calculates average score across
    k folds
    '''
    scores = cross_validate(model, 
                            X, 
                            z, 
                            k_folds, 
                            X_scaler, 
                            z_scaler,
                            metrics)
    score_dict = {}
    if (metrics is not None):
        score_dict={'train':{},'test':{}}
        for metric in metrics:
            #Calculates the mean score from the cross validation
            #and adds train and test metrics(MSE,R2 etc) to score_dict.
            score_dict['train'].update({metric:mean(scores['train_scores'][metric])})
            score_dict['test'].update({metric:mean(scores['test_scores'][metric])})
        return score_dict
          
    return mean(scores['train_scores']), mean(scores['test_scores'])