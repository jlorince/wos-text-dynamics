import numpy as np

def cond_median(arr_x,arr_y,arr_bins,y_perc = [25,75]):
    x_med = []
    y_med = []
    y_err = []
    arr_x = np.array(arr_x)
    arr_y = np.array(arr_y)
    for i_bin in range(len(arr_bins)-1):
        x1 = arr_bins[i_bin]
        x2 = arr_bins[i_bin+1]
        inds = np.where( (arr_x>=x1)*(arr_x<x2) )[0]
        x_tmp = arr_x[inds]
        y_tmp = arr_y[inds]
        if len(y_tmp)>0:
            x_med += [np.median(x_tmp)]
            y_med += [np.median(y_tmp)]
            y_err += [np.percentile(y_tmp,y_perc)]
    y_err_arr = np.zeros((2,len(y_err)))
    for i_y_,y_ in enumerate(y_err):
        y_err_arr[:,i_y_] = np.abs(y_med[i_y_] - y_)
    return np.array(x_med),np.array(y_med),np.array(y_err_arr)


def running_median(arr_x,arr_y,n_run = 100,y_perc = [25,75]):
    x_med = []
    y_med = []
    y_err = []
    
    ## sort first according to x keeping relation between x and y
    arr_x = np.array(arr_x)
    arr_y = np.array(arr_y)
    indsort = np.argsort(arr_x)
    arr_x = arr_x[indsort]
    arr_y = arr_y[indsort]
    
    N_ = len(arr_x)
    if int(N_/n_run) == 0:
        n_run = n_run/10
    if N_%n_run == 0:
        s_ = int(N_/float(n_run))
    else:
        s_ = int(N_/float(n_run))+1

        
    for i_s in range(n_run):
        if i_s < s_ - 1:
            x_tmp = arr_x[i_s*s_:(i_s+1)*s_]
            y_tmp = arr_y[i_s*s_:(i_s+1)*s_]
        else:
            x_tmp = arr_x[i_s*s_:]
            y_tmp = arr_y[i_s*s_:]

        x_med += [np.median(x_tmp)]
        y_med += [np.median(y_tmp)]
        y_err += [np.percentile(y_tmp,y_perc)]
    
    y_err_arr = np.zeros((2,len(y_err)))
    for i_y_,y_ in enumerate(y_err):
        y_err_arr[:,i_y_] = np.abs(y_med[i_y_] - y_)
    return np.array(x_med),np.array(y_med),np.array(y_err_arr)

def running_median_long(arr_x,arr_y,nw = 100,y_perc = [25,75]):
    x_med = []
    y_med = []
    y_err = []
    
    ## sort first according to x keeping relation between x and y
    indsort = np.argsort(arr_x)
    arr_x = arr_x[indsort]
    arr_y = arr_y[indsort]
    
    N_ = len(arr_x)
        
    for i_s in range(N_ - nw):

        x_tmp = arr_x[i_s:i_s+nw]
        y_tmp = arr_y[i_s:i_s+nw]

        x_med += [np.median(x_tmp)]
        y_med += [np.median(y_tmp)]
        y_err += [np.percentile(y_tmp,y_perc)]
    
    y_err_arr = np.zeros((2,len(y_err)))
    for i_y_,y_ in enumerate(y_err):
        y_err_arr[:,i_y_] = np.abs(y_med[i_y_] - y_)
    return np.array(x_med),np.array(y_med),np.array(y_err_arr)

### correlations
def C_running_median(arr_x,arr_y,n_run = 100):
    x_med = []
    y_med = []
    C_med = []
    arr_x = np.array(arr_x)
    arr_y = np.array(arr_y)
    ## sort first according to x keeping relation between x and y
    indsort = np.argsort(arr_x)
    arr_x = arr_x[indsort]
    arr_y = arr_y[indsort]
    
    N_ = len(arr_x)
    if int(N_/n_run) == 0:
        n_run = n_run/10
    if N_%n_run == 0:
        s_ = int(N_/float(n_run))
    else:
        s_ = int(N_/float(n_run))+1

    print(n_run)
    for i_s in range(n_run):
        if i_s < s_ - 1:
            x_tmp = arr_x[i_s*s_:(i_s+1)*s_]
            y_tmp = arr_y[i_s*s_:(i_s+1)*s_]
        else:
            x_tmp = arr_x[i_s*s_:]
            y_tmp = arr_y[i_s*s_:]

        x_med += [np.median(x_tmp)]
        y_med += [np.median(y_tmp)]
        C_med += [spearmanr(x_tmp,y_tmp)]
    

    return np.array(x_med),np.array(y_med),np.array(C_med)