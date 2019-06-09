import math
import numpy as np
import random
from sklearn.preprocessing import minmax_scale


def norm_squared(X):
    return np.square(np.linalg.norm(X,axis=1))

def normalize(data):
    return minmax_scale(data)

def hz2mel(hz):
    mel=2595*np.log10(1+hz/700)
    return mel


def mel2hz(mel):
    hz=700*(10**(mel/2595)-1)
    return hz