from numpy import sum,mean,size

def R2(target, prediction):
    '''
    Returns the R2 score of prediction compared
    to target.
    '''
    return 1 - sum((target - prediction) ** 2) / sum((target - mean(target)) ** 2)

def MSE(target, prediction):
    '''
    Returns the MSE of prediction compared 
    to target.
    '''
    
    return ((target-prediction)**2).mean()

def MSE_R2(target, prediction):
    '''
    Returns MSE and R2
    '''
    return MSE(target,prediction),R2(target,prediction)

def accuracy(target, prediction):
    '''
    Returns the accuracy score of prediction compared
    to target.
    '''
    return sum(prediction==target)/len(target)

METRIC_FUNC = {'R2':R2,
          'MSE':MSE,
          'accuracy':accuracy}

def scores(target,prediction):
    '''
    Returns the MSE and R2 score for target and prediction
    '''
    return MSE(target,prediction),R2(target,prediction)

