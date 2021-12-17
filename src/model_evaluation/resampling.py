from sklearn.model_selection import StratifiedKFold,KFold,train_test_split
from sklearn.utils import resample
from numpy import mean, empty, var, zeros, squeeze, asarray, expand_dims,square
from sklearn.preprocessing import LabelBinarizer
from .metrics import METRIC_FUNC

def bootstrap(model, X_train, X_test, z_train,n_bootstraps=200):

    z_pred = empty((X_test.shape[0], 6,n_bootstraps))   
    for i in range(n_bootstraps):
        X_,z_ = resample(X_train, z_train)         #random resampling 
        model.fit(X_,z_)                #fit beta to new resampled data
        z_pred[:,:,i] = model.predict_proba(X_test) #predict test data

    return z_pred

def bias_var(model, X_train,X_test, z_train,z_test, n_bootstraps=200):
    
    z_pred = bootstrap(model,X_train, X_test, z_train,n_bootstraps)  
    
    bias = squeeze(mean( square(expand_dims(z_test,axis=2) - mean(z_pred, axis=2,keepdims=True)),keepdims=True,axis=0))
    variance = squeeze(mean( var(z_pred, axis=2, keepdims=True), axis=0, keepdims=True))
    for i in range(z_pred.shape[-1]):
        z_pred[:,:,i] = z_pred[:,:,i]-z_test 
    error = squeeze(mean( mean(square(z_pred), axis=2, keepdims=True),axis=0 , keepdims=True))
    
    
    return error, bias, variance

def bias_var_analysis(model, X_train,X_test, z_train,z_test,comp_param,n_bootstraps=200): 
    '''
    Calculates the error, bias and variance for every class in a multiclass model output.
    Naively uses mse as loss.
    
    Inputs:
    -------
    model: classifier
    
    X_train: train features
    
    X_test: test features
    
    z_train: labels with shape(n_samples,), the labels are either encoded or strings
    
    z_test: labels with shape(n_samples,), the labels are either encoded or strings
    
    com_param: dict{parameter_name:values}
                complexity parameter instance tree depth.
                
    n_bootstraps: number of bootstraps
    
    
    Output:
    -------
    error: numpy array shape(n_params,n_classes), error for each class
    
    bias: numpy array shape(n_params,n_classes), bias(squared) for each class
    
    var: numpy array shape(n_params,n_classes), variance for each class
    '''
    #Converting to numpy
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    z_train = z_train.to_numpy()
    z_test = z_test.to_numpy()
    
    #Must binarize test set as model output is binarized.
    #and they need to be same shape.
    enc = LabelBinarizer().fit(z_train.reshape(-1,1))
    z_test = enc.transform(z_test)
    
    
    if (len(comp_param)>1):
        print("'comp_param' can only contain one parameter")
        raise ValueError
    
    #Number of parameters to test
    n_params = len(list(comp_param.values())[0])
    #Number of classes
    n_classes = z_test.shape[-1]
    #Name of parameter
    param_str = list(comp_param.keys())[0]
    
    #Preallocating arrays
    error = zeros((n_params,n_classes))
    bias = zeros((n_params,n_classes))
    variance = zeros((n_params,n_classes))

       
    for i,val in enumerate(list(comp_param.values())[0]):
        model.set_params(**{param_str:val})
        #Calculate error, bias and variance for model i with complexity val
        error[i,:], bias[i,:], variance[i,:] = bias_var(model, 
                                                  X_train, 
                                                  X_test, 
                                                  z_train,
                                                  z_test,
                                                  n_bootstraps=n_bootstraps)
    return error,bias, variance

def cross_validate(model, X, z, k_folds=4, X_scaler=None, z_scaler = None, metrics = None):
    '''
    Calculates the cross validated score of model.
    The scoring metric  is given by the model itself.
    Ex. 
    model is a classifier -> metric is accuracy.
    model is a regressor -> metric is R2.
    '''

    #Note that due to class imbalance a stratifiedkfold must be used as well.
    kfold = StratifiedKFold(n_splits = k_folds, shuffle=False)

    if (metrics is not None):
        train_scores = {}
        test_scores = {}
        for metric in metrics:
            train_scores[metric] = list()
            test_scores[metric] = list()
        
    else:
        train_scores = []
        test_scores = []
    
    for train_inds, test_inds in kfold.split(X,z):
        X_train = X[train_inds]
        z_train = z[train_inds]

        X_test = X[test_inds]
        z_test = z[test_inds]
        
        #Scale data if scalers are passed
        if(X_scaler is not None):
            X_train = X_scaler.fit_transform(X_train)
            X_test = X_scaler.transform(X_test)
        if(z_scaler is not None):
            z_train = z_scaler.fit_transform(z_train.reshape(-1,1))
            z_test = z_scaler.transform(z_test.reshape(-1,1)) 
        
        #Fit the model
        model.fit(X_train, z_train)
        if(metrics is not None):
            p_train = model.predict(X_train)
            p_test = model.predict(X_test)
            for metric in metrics:
                if(metric in ['recall','precision','f1']):
                    train_scores[metric].append(METRIC_FUNC[metric](z_train,p_train.reshape(-1,1),average='macro',zero_division=0))
                    test_scores[metric].append(METRIC_FUNC[metric](z_test,p_test.reshape(-1,1),average='macro',zero_division=0))
                else:
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