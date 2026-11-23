import numpy as np

def get_main_frequency_zcr(signal, sampling_rate):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    num_crossings = len(zero_crossings)
    duration = len(signal) / sampling_rate
    return (num_crossings / 2) / duration

