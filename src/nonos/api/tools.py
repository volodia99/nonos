import numpy as np


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_around(array, value):
    array = np.asarray(array)
    idx_1 = (np.abs(array - value)).argmin()
    larray = list(array)
    larray.remove(larray[idx_1])
    arraym = np.asarray(larray)
    idx_2 = (np.abs(arraym - value)).argmin()
    return np.asarray([array[idx_1], arraym[idx_2]])
