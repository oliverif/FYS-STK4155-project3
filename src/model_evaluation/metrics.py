from numpy import sum,mean,size, insert, vstack,asarray
from sklearn.metrics import balanced_accuracy_score, confusion_matrix,precision_recall_fscore_support,accuracy_score,f1_score,precision_score,recall_score
import pandas as pd

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
          'accuracy':accuracy,
          'precision':precision_score,
          'recall':recall_score,
          'f1':f1_score}

def scores(target,prediction,classes,metrics):
    '''
    Returns a collection of classification scores
    in a DataFrame
    '''
    #Confusion matrix of predictions.
    cm = confusion_matrix(target, prediction)
    #Compute class accuracies
    #by dividing diagonal of cm(which contains correct predictions) 
    # with sum of all(i.e. n samples), one obtains the classwise accuracy
    #Compute class precision
    class_scores =  insert(asarray(precision_recall_fscore_support(target,prediction,zero_division=0)).T,0,
                          cm.diagonal()/cm.sum(axis=1),axis=1)  
    #Macro scores, i.e. averaged with equal weight               
    macro_scores = insert(asarray(precision_recall_fscore_support(target,prediction,average='macro',zero_division=0)).T,0,
                            balanced_accuracy_score(target,prediction))
    #Micro scores, i.e. simply count tp, fp, tn and fn. Note that all of these are always the same and
    # need not really be calculated. It's sufficient with one, and then repeat, however is left there
    # for demonstration purposes.
    micro_scores = insert(asarray(precision_recall_fscore_support(target,prediction,average='micro',zero_division=0)).T,0,
                            accuracy_score(target,prediction))
    return pd.DataFrame(vstack((class_scores,macro_scores,micro_scores)),index = classes, columns=metrics)

