import numpy as np
import copy
import metrics
        
def compute_metrics(y_pred, y_true):
    
    '''
    Given predictions and labels, compute a few metrics.
    '''
    
    num_examples, num_classes = np.shape(y_true)
    
    results = {}
    average_precision_list = []
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        average_precision_list.append(metrics.compute_avg_precision(y_true[:, j], y_pred[:, j]))
    hamming = metrics.compute_hamming_loss(y_true, y_pred)
        
        
    results['map'] = 100.0 * float(np.mean(average_precision_list))
    results['ap'] = 100.0 * np.array(average_precision_list)
    results['hamming'] = 100.0 * hamming
    
    return results