from numpy import zeros
from sklearn.model_selection import GridSearchCV
import pandas as pd
from .resampling import cross_val_score
import timeit



def evaluate_parameter(model, params, metrics, X, z,X_scaler,z_scaler):
    '''
    Fits a model using for every 
    'param' value in 'vals'.
    Returns MSE and/or R2 score arrays for the different
    parameter values.
    
    Inputs
    -----------
    param: {'lambda', 'lr', 'n_epochs','n_batches'}
        Parameter to evaluate

    vals: ndarray of shape (n_params,)
        Array containing different values of param to evaluate

    model: model 
        A LinReg,SGD_linreg,NeuralNetwork)
        SGD optimizer object used to fit weights with above parameter
        
    Output:
    -------
    score_dict: dict{metric:{'train':scores, 'test':scores}}
        Nested dictionary containing the metrics for the parameters.
        I.e each outer value is the metric name, which then contains
        dictionary with train and test scores. The scores has length
        
    '''
    param_name = list(params.keys())[0]

    #Number of parameters to test
    num_vals = params[param_name].shape[0]

    #Arrays to store scores
    score_dict = {}
    score_dict= dict((metric,{'train':[],'test':[]}) for metric in metrics)
    score_dict = dict((metric,{'train':[],'test':[]}) for metric in metrics)
    score_dict['train_time'] = []



    #Fit with every val and store metrics
    for i in range(num_vals):
        #Use dict with param_name as key and element i from
        #vals as value to set model parameter.
        model.set_params(**{param_name:params[param_name][i]})
        start = timeit.default_timer()
        scores = cross_val_score(model, X, z,5,X_scaler,z_scaler,metrics)
        stop = timeit.default_timer()
        for metric in metrics:
            score_dict[metric]['train'].append(scores['train'][metric])
            score_dict[metric]['test'].append(scores['test'][metric])
        score_dict['train_time'].append(stop - start)

    return score_dict

    
def grid_search_df(X, z, model, param_grid):
    '''
    Performs a grid search for best model
    performance across param_grid. 
    Function wraps sklearn GridSearchCV and
    outputs more readable results. SKlearn implements
    multiprocessing and enables much faster execution.
    

    Inputs:
    -------
    X: ndarray(n_samples,n_features)
        Design matrix
        
    z: ndarray(n_samples,1)
        Target data
    
    model: model object
        The model to perform the gridsearch for.
        
    param_grid: dict(param_name=list_of_values)
        Dictionary containing parameter names and the
        values to test for. Note that param_name
        must match exactly the parameter name of the
        model. This dictionary is passed further to
        GridSearchCV
        
    Outputs:
    -------
    gs: GridSearch object
        A model fitted on the best parameters found
        in gridsearch.
    
    df: pandas.DataFrame
        Dataframe containing the results of the gridsearch.
    '''
    #Gridsearch object
    gs = GridSearchCV(estimator = model, 
                      cv=5,
                      param_grid = param_grid,
                      n_jobs=1)
    #Fit the model
    gs = gs.fit(X,z)
    #Create list of parameters as presented in the GridSearchCV
    #output
    param_strs =['param_'+s for s in list(param_grid.keys())]
    #Extract only the useful columns
    data = {k[6:]: gs.cv_results_[k] for k in (param_strs)}
    data['mean_test_score'] = gs.cv_results_['mean_test_score']
    data['rank_test_score'] = gs.cv_results_['rank_test_score']

    return gs, pd.DataFrame(data)