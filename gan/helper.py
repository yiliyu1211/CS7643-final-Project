from skimage.metrics import peak_signal_noise_ratio
import numpy as np

def nmse(pred, y):
    """
    NMSE
    input: y, pred should be in the same shape (any shape should work)
    """
    res = np.linalg.norm(y-pred)**2 / np.linalg.norm(y)**2
    return res 


def psnr(pred, y):
    """
    peak signal noise ratio 
    input: y, pred should be in the same shape (any shape should work)
    """
    dmax, dmin = np.max(y), np.min(y)
    drange = dmax - dmin
    # pred_norm = (pred-dmin)/range
    # y_norm = (y-dmin)/range
    res = peak_signal_noise_ratio(y, pred, data_range=drange)

    return res
