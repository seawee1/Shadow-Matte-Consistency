import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sigmoid(x, alpha, beta):
    return 1.0/(1.0 + np.exp(-alpha*(x-beta)))

def fitSigmoid(patch):
    # Patch mean computation
    # Compute mean over y-axis
    mean = np.sum(patch, axis=0)
    mean = mean/patch.shape[0]
    # Compute mean over color channels
    #mean = np.sum(mean, axis=1)
    #mean = mean/3.0
    
    #Normalize, so that in [0.0, 1.0]
    mean = mean - np.min(mean)
    mean = mean / np.max(mean)
    
    x = np.arange(0, patch.shape[1], step=1)
    plt.plot(x, mean)
    
    popt, pcov = curve_fit(sigmoid, x, mean)
    plt.plot(x, sigmoid(x, popt[0], popt[1]))
    return popt

def penumbraRegion(patch):
    params = fitSigmoid(patch)
    alpha = params[0]
    beta = params[1]
    
    beg = -2.0 * 1/(alpha * np.sqrt(np.pi/8.0)) + beta
    plt.axvline(x=beg, color='red')
    end = 2.0 * 1/(alpha * np.sqrt(np.pi/8.0)) + beta
    plt.axvline(x=end, color='red')
    return np.array([beg, end])