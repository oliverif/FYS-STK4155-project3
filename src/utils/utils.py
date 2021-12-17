def sort_surface(X,z):
    '''
    Sorts the design matrix and correspondix targed 
    data in a Radiz sort fashion.
    First sort by x-column, then sort by y-column.
    'Stable' sort must be used to obtain correct order.
    '''
    if(X.shape[1]!=2):
        p1 = X[:,1].argsort(kind='stable')   
    else:
        p1 = X[:,0].argsort(kind='stable')   
    X = X[p1]
    z = z[p1]

    if(X.shape[1]!=2):
        p2 = X[:,2].argsort(kind='stable')  
    else:
        p2 = X[:,1].argsort(kind='stable') 
    
    X = X[p2]
    z = z[p2]
    return X,z