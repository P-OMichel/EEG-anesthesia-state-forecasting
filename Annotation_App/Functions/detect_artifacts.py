'''
Detects the artifacts in the EEG signal using thresholds on wavelet coefficients CDF slopes.
ref. Jiaqi Wang master thesis: "Real-Time EEG artifact detection for continuous Monitoring during anesthesia"
'''
'''
minimize the loss of clean eeg using the slopes (which size of window to use ans step according to the size of an artifact to avoid computing slopes with a too large mix of clean and artifacted signal)
see if sliding mean on abs value of the signal can be good for detection
'''

import numpy as np
import pywt         # https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for list of wavelet family name

from Annotation_App.Functions.ecdf import ecdf
from scipy import stats
from Annotation_App.Functions.filter import filter_butterworth
import scipy as sc
from Annotation_App.Functions.utils import detect_pos_1
from pybaselines import Baseline
import Annotation_App.Functions.time_frequency as tf
from Annotation_App.Functions.utils import diff_envelops
#from sklearn.neighbors import LocalOutlierFactor
from Annotation_App.Functions.detect_burst import get_burst
from Annotation_App.Functions.utils import resize_binary_mask
from scipy.signal import find_peaks, peak_prominences

def correct_peaks(y, fs, f_cut, threshold):

    N = len(y)

    lf_signal = filter_butterworth(y,fs,[0.1,f_cut])
    hf_signal = filter_butterworth(y,fs,[f_cut,45])

    # Find peaks and compute their properties
    peaks, properties = sc.signal.find_peaks(lf_signal, 
                                          width=[8,32]
                                          )
    
    prominences = properties['prominences']
    widths = properties['widths']
    
    # fixed threshold value
    prominence_threshold = threshold 
    
    # Detect peaks that satisfy the conditions
    relevant_peaks = []
    widths_peaks = []
    for i, peak in enumerate(peaks):
        if prominences[i] > prominence_threshold:
            relevant_peaks.append(peak)
            widths_peaks.append(widths[i])

    mask_peaks=np.zeros(N)
    for i in range(len(relevant_peaks)):
        start = int(relevant_peaks[i] - widths_peaks[i])
        end = int(relevant_peaks[i] + widths_peaks[i])

        if start < 0:
            start = 0
        if end >= N:
            end = N - 1
    
        mask_peaks[start:end]=1

    # convert mask to list of start and end position of artefacts
    pos_peaks = detect_pos_1(mask_peaks)

    for pos in pos_peaks:
        x= range(pos[0]+1,pos[-1])
        y[pos[0]+1:pos[-1]] = np.interp(x,[pos[0],pos[-1]],[lf_signal[pos[0]],lf_signal[pos[-1]]]) + hf_signal[pos[0]+1:pos[-1]]
    
    return y, mask_peaks

'''
def find_artifacts(y,
                   Ws,
                   step,
                   threshold,
                   wavelet_name,
                   level,
                   mode):
    
    Inputs:
    - y              <-- raw eeg signal
    - Ws             <-- sliding window size on wich the DWD is applied
    - step           <-- step size between windows
    - threshold      <-- threshold for the slopes (list of 2, one for the approximation coefficient slope and the other for the first detail coefficient slope)
    - wavelet name   <-- name of the wavelet family to use
    - level          <-- number of detail coefficients
    - mode           <-- mode to use for DWT
    
    Outputs:
    - mask_artifacts <-- 1 at the position where there is no artifacts and 0 where there is one
    - index_mask     <-- marks the position of the iteration each time there is a change in 1/0
    

    N=len(y)      # length of the EEG signal that is sent as input (should be the same as the batch of data gathered from the device before doing compution an ideally should be a multiple of Ws)

    # gathering the slopes, thresholding and creating mask

    ## initialization
    start_window=0                              # marker of the position of the left window edge for the next computation

    # creating the mask    
    threshold_a,threshold_d1=threshold          # receives the thresholds value
    mask_artifacts=[1 for i in range(N)]        # initialize the mask
    mask_pos=[]                                 # set iteration indexes of where the artefacts start and end. (useful for Clean_EEG.py to avoid an unecessary search index in a list to get positions of 0 in the mask) 
    
    while start_window<=N:
        if start_window+Ws<=N:                           # checking that the marker for next computation + the size of the window is within the length of the signal

            s_a,s_d1=CDF_Slope(y[start_window:start_window+Ws],wavelet_name,level,mode)
        
            if (s_a<threshold_a or s_d1<threshold_d1) :  # slope lower than threshold_a indicating an artifact EOG or Motion and lower than threshold_d1 indicating an artifact EMG
                mask_artifacts[start_window:start_window+Ws]=[0 for i in range(Ws)]
                mask_pos.append([start_window,start_window+Ws]) # in case
            start_window+=step                           # will go above n in case of equality and stops the cycle

        elif (start_window<N and start_window+Ws>N):     # still within the signal but next window will have a part in the signal and another outside. si on prend plus de signal (au moin step en iteration de plus) ne pose plus de pblm (voir avec pour le spectro)
            
            s_a,s_d1=CDF_Slope(y[start_window:],wavelet_name,level,mode)

            if (s_a<threshold_a or s_d1<threshold_d1):    
                mask_artifacts[start_window:]=[0 for i in range(N-start_window)]
                mask_pos.append([start_window,N-1]) # in case
            start_window+=Ws                             # will be higher than N and this stops the cycle
    

    if len(mask_pos)<1:
        #print('no artifact detected')
        return []
    else:
        index_mask=[mask_pos[0][0]]
    for i in range(len(mask_pos)-1):
        if np.abs(mask_pos[i][1]-mask_pos[i+1][0])>step:
            index_mask.append(mask_pos[i][1])
            index_mask.append(mask_pos[i+1][0])
    index_mask.append(mask_pos[-1][1])

    return index_mask
'''

def is_outlier_iqr(data):
    Q1 = np.percentile(data, 30) # 30
    Q3 = np.percentile(data, 70) # 70
    IQR = Q3 - Q1
    T = 10                       # 10
    lower_bound = Q1 - T * IQR
    upper_bound = Q3 + T * IQR
    return (data > upper_bound), upper_bound #(data < lower_bound) | (data > upper_bound)

# def is_outlier_diff_quantile(data):
#     Q1 = np.percentile(data, 50)
#     Q3 = np.percentile(data, 70)
#     IQR = Q3 - Q1
#     T = 10
#     threshold = T * IQR
#     return (data > threshold), threshold

def is_outlier_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

def remove_short_art(signal,pos_art,ws,smoothing_ws):
    '''
    ws: number of neighbours to take on the left and right
    smoothing_ws: number of points to take for smoothing (associated to a cutting frequency)
    '''

    N=len(signal)
    fit = np.zeros(N)
    for pos in pos_art:

        i = pos[0] - ws
        j = pos[-1] + ws

        if i < 0:
            i = 0
        if j >=N:
            j = N - 1

        #signal[i:j] = signal[i:j] - sc.signal.savgol_filter(signal[i:j],smoothing_ws,1)
        fit[i:j] = sc.signal.savgol_filter(signal[i:j],smoothing_ws,1)

        # add polynomial fit
        x = np.array([i , j])
        y = np.array([signal[i], signal[j]])

        # Fit a quadratic (parabola) curve to these two points
        coeff = np.polyfit(x, y, 2)
        poly = np.poly1d(coeff)

        # Use the polynomial to estimate the missing value at index 'i'
        x_interp = range(i,j)
        fit[i:j] = fit[i:j] + poly(x_interp)  

    return fit

# OTE: artifacts with sliding windows
#---------------------------------------------------------------------------------------------

def neighbor_median(x: np.ndarray, radius: int) -> np.ndarray:
    """
    Rolling median with a symmetric neighborhood of size (2*radius + 1).
    Edges use the available neighbors (smaller window).
    """
    x = np.asarray(x)
    n = x.size
    out = np.empty(n, dtype=float)

    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out[i] = np.median(x[lo:hi])

    return out

def cdf_wave_artifacts(
    y,
    Ws,
    step,
    threshold,
    wavelet_name,
    level,
    mode,
    median_mode="global",       # "global" or "neighbors"
    neighbor_radius=10,         # only used if median_mode="neighbors"
    median_divisor=10.0         # your "/ 10"
):
    """
    Outputs:
    - mask_a, mask_d1: True where "no artifact" (or whatever your convention is)
    - t_list
    - slopes_list (2 x n_windows) presumably: [approx_slope, detail1_slope]
    """
    N = len(y)
    pos = np.arange(0, N)
    windows = [y[i:i+Ws] for i in range(0, N - Ws, step)]
    pos_list = [pos[Ws + i*step] for i in range(len(windows))]

    threshold_a, threshold_d1 = threshold

    slopes_list = np.transpose([
        CDF_Slope(win, wavelet_name, level, mode) for win in windows
    ])  # shape: (2, n_windows)

    a_slopes = slopes_list[0]
    d1_slopes = slopes_list[1]

    if median_mode == "global":
        a_med = np.median(a_slopes)
        d1_med = np.median(d1_slopes)
        a_ref = np.full_like(a_slopes, a_med, dtype=float)
        d1_ref = np.full_like(d1_slopes, d1_med, dtype=float)

        mask_a = (a_slopes <= threshold_a) & (a_slopes <= (a_ref / median_divisor))
        mask_d1 = (d1_slopes <= threshold_d1) & (d1_slopes <= (d1_ref / median_divisor))

    elif median_mode == "neighbors":
        a_ref = neighbor_median(a_slopes, radius=neighbor_radius)
        d1_ref = neighbor_median(d1_slopes, radius=neighbor_radius)
        
        mask_a = (a_slopes <= threshold_a) & (a_slopes <= (a_ref / median_divisor))
        mask_d1 = (d1_slopes <= threshold_d1) & (d1_slopes <= (d1_ref / median_divisor))

    elif median_mode == None:
    
        mask_a = (a_slopes <= threshold_a) 
        mask_d1 = (d1_slopes <= threshold_d1) 

    else:
        raise ValueError('median_mode must be "global" or "neighbors"')

    return mask_a, mask_d1, slopes_list, pos_list

def merge_and_dilate(mask1, mask2, dilation=1):
    """
    mask1, mask2: 1D numpy arrays of 0/1, same length
    dilation: number of samples to dilate on each side
    """
    # Ensure numpy arrays
    mask1 = np.asarray(mask1, dtype=np.uint8)
    mask2 = np.asarray(mask2, dtype=np.uint8)

    # Step 1: OR merge
    merged = np.logical_or(mask1, mask2).astype(np.uint8)

    # Step 2: dilation
    kernel = np.ones(2 * dilation + 1, dtype=np.uint8)
    dilated = np.convolve(merged, kernel, mode="same") > 0

    return dilated.astype(np.uint8)

# def transition_indices(mask):
#     mask = np.asarray(mask, dtype=int)
#     padded = np.pad(mask, (1, 1), constant_values=0)
#     return np.where(np.diff(padded) != 0)[0].tolist()
def transition_indices(mask):
    m = np.asarray(mask, dtype=bool)
    n = m.size
    if n == 0:
        return []

    # starts: indices where 0->1 happens (run starts)
    starts = np.where(~m[:-1] & m[1:])[0] + 1
    # ends: indices where 1->0 happens, but we want the last 1 index (inclusive)
    ends = np.where(m[:-1] & ~m[1:])[0]

    # handle boundary runs
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, n - 1]

    # interleave start/end: [s0,e0,s1,e1,...]
    out = np.empty(starts.size + ends.size, dtype=int)
    out[0::2] = starts
    out[1::2] = ends
    return out.tolist()

def get_artifacts_and_gc(y,
                   Ws,
                   step,
                   threshold,
                   wavelet_name,
                   level,
                   mode):
    
    mask_a, mask_d1, slopes_list, pos_list = cdf_wave_artifacts(y, 
                                                                Ws,
                                                                step,
                                                                threshold,
                                                                wavelet_name,
                                                                level,
                                                                mode, 
                                                                median_mode='neighbors', 
                                                                neighbor_radius=7, 
                                                                median_divisor=3.0 )
    
    mask = merge_and_dilate(mask_a, mask_d1, dilation = 1)

    # gc
    low_neighbor = (slopes_list[0] <= 0.003) | (slopes_list[1] <= 0.03)
    size = 6  # neighborhood radius
    kernel = np.ones(2 * size + 1, dtype=int)

    neighbor_exists = np.convolve(
        low_neighbor.astype(int),
        kernel,
        mode="same"
    ) > 0

    mask_gc = (slopes_list[0] >= 0.1) & neighbor_exists

    prop_gc = np.sum(mask_gc) / len(mask_gc)

    pos_list = np.array(pos_list)
    pos_list = list(pos_list[transition_indices(mask)])   

    return mask, mask_gc, prop_gc, pos_list#, slopes_list # remove slopes_list for real time


#-----------------------------------------------------------------------------------------------------------

def find_artifacts(y,
                   Ws,
                   step,
                   threshold,
                   wavelet_name,
                   level,
                   mode):
    '''
    Inputs:
    - y              <-- raw eeg signal
    - Ws             <-- sliding window size on wich the DWD is applied
    - step           <-- step size between windows
    - threshold      <-- threshold for the slopes (list of 2, one for the approximation coefficient slope and the other for the first detail coefficient slope)
    - wavelet name   <-- name of the wavelet family to use
    - level          <-- number of detail coefficients
    - mode           <-- mode to use for DWT
    
    Outputs:
    - mask_artifacts <-- 1 at the position where there is no artifacts and 0 where there is one
    - index_mask     <-- marks the position of the iteration each time there is a change in 1/0
    '''

    N=len(y)      # length of the EEG signal that is sent as input (should be the same as the batch of data gathered from the device before doing compution an ideally should be a multiple of Ws)

    # gathering the slopes, thresholding and creating mask

    ## initialization
    start_window=0                              # marker of the position of the left window edge for the next computation

    # creating the mask    
    threshold_a,threshold_d1=threshold          # receives the thresholds value
    mask_artifacts=[1 for i in range(N)]        # initialize the mask
    mask_pos=[]                                 # set iteration indexes of where the artefacts start and end. (useful for Clean_EEG.py to avoid an unecessary search index in a list to get positions of 0 in the mask) 
    
    while start_window<=N:
        if start_window+Ws<=N:                           # checking that the marker for next computation + the size of the window is within the length of the signal

            s_a,s_d1=CDF_Slope(y[start_window:start_window+Ws],wavelet_name,level,mode)
            #med_amp=np.quantile(np.abs(y[start_window:start_window+Ws]),1)
        
            if (s_a<threshold_a or s_d1<threshold_d1):# or med_amp>=90) :  # slope lower than threshold_a indicating an artifact EOG or Motion and lower than threshold_d1 indicating an artifact EMG
                mask_artifacts[start_window:start_window+Ws]=[0 for i in range(Ws)]
                mask_pos.append([start_window,start_window+Ws]) # in case
            start_window+=step                           # will go above n in case of equality and stops the cycle

        elif (start_window<N and start_window+Ws>N):     # still within the signal but next window will have a part in the signal and another outside. si on prend plus de signal (au moin step en iteration de plus) ne pose plus de pblm (voir avec pour le spectro)
            
            s_a,s_d1=CDF_Slope(y[start_window:],wavelet_name,level,mode)

            if (s_a<threshold_a or s_d1<threshold_d1): # or med_amp>=90):    
                mask_artifacts[start_window:]=[0 for i in range(N-start_window)]
                mask_pos.append([start_window,N-1]) # in case
            start_window+=Ws                             # will be higher than N and this stops the cycle
    

    if len(mask_pos)<1:
        #print('no artifact detected')
        return []
    else:
        index_mask=[mask_pos[0][0]]
    for i in range(len(mask_pos)-1):
        if np.abs(mask_pos[i][1]-mask_pos[i+1][0])>step:
            index_mask.append(mask_pos[i][1])
            index_mask.append(mask_pos[i+1][0])
    index_mask.append(mask_pos[-1][1])

    return index_mask

def find_artifacts_mask(y,
                   Ws,
                   step,
                   threshold,
                   wavelet_name,
                   level,
                   mode):
    '''
    Inputs:
    - y              <-- raw eeg signal
    - Ws             <-- sliding window size on wich the DWD is applied
    - step           <-- step size between windows
    - threshold      <-- threshold for the slopes (list of 2, one for the approximation coefficient slope and the other for the first detail coefficient slope)
    - wavelet name   <-- name of the wavelet family to use
    - level          <-- number of detail coefficients
    - mode           <-- mode to use for DWT
    
    Outputs:
    - mask_artifacts <-- 1 at the position where there is no artifacts and 0 where there is one
    - index_mask     <-- marks the position of the iteration each time there is a change in 1/0
    '''

    N=len(y)      # length of the EEG signal that is sent as input (should be the same as the batch of data gathered from the device before doing compution an ideally should be a multiple of Ws)

    # gathering the slopes, thresholding and creating mask

    ## initialization
    start_window=0                              # marker of the position of the left window edge for the next computation

    # creating the mask    
    threshold_a,threshold_d1=threshold          # receives the thresholds value
    mask_artifacts=[1 for i in range(N)]        # initialize the mask
    mask_pos=[]                                 # set iteration indexes of where the artefacts start and end. (useful for Clean_EEG.py to avoid an unecessary search index in a list to get positions of 0 in the mask) 
    
    while start_window<=N:
        if start_window+Ws<=N:                           # checking that the marker for next computation + the size of the window is within the length of the signal

            s_a,s_d1=CDF_Slope(y[start_window:start_window+Ws],wavelet_name,level,mode)
            #med_amp=np.quantile(np.abs(y[start_window:start_window+Ws]),1)
        
            if (s_a<threshold_a or s_d1<threshold_d1):# or med_amp>=90) :  # slope lower than threshold_a indicating an artifact EOG or Motion and lower than threshold_d1 indicating an artifact EMG
                mask_artifacts[start_window:start_window+Ws]=[0 for i in range(Ws)]
                mask_pos.append([start_window,start_window+Ws]) # in case
            start_window+=step                           # will go above n in case of equality and stops the cycle

        elif (start_window<N and start_window+Ws>N):     # still within the signal but next window will have a part in the signal and another outside. si on prend plus de signal (au moin step en iteration de plus) ne pose plus de pblm (voir avec pour le spectro)
            
            s_a,s_d1=CDF_Slope(y[start_window:],wavelet_name,level,mode)

            if (s_a<threshold_a or s_d1<threshold_d1): # or med_amp>=90):    
                mask_artifacts[start_window:]=[0 for i in range(N-start_window)]
                mask_pos.append([start_window,N-1]) # in case
            start_window+=Ws                             # will be higher than N and this stops the cycle
    
        print(s_a, s_d1, start_window)
    return mask_artifacts

def CDF_Slope(y,wavelet_name,level,mode):
    '''
    Inputs: 
    - y            <-- part of the eeg signal on which the DWD is applied
    - wavelet name <-- name of the wavelet family to use
    - level        <-- number of detail coefficients
    - mode         <-- mode to use for DWT

    Outputs:
    - CDFa         <-- CDF of the approximation coefficient (if wanted)
    - CDFd1        <-- CDF of the first detail coefficient  (if wanted)
    - slope_a      <-- Slope joining the first and last point of CFDa 
    - slope_d1     <-- Slope joining the first and last point of CDFd1
    '''

    # Compute the first detail coefficient and the approximation one (using DWD)
    ca,cd1=coeffs_wavelet(y,wavelet_name,level,mode)

    # Compute the empirical CDF for each coefficient
    CDFa=ecdf(np.abs(ca)) # CDFa[0] is the list of values appearing in ca and CDFa[1] is the list of associated cumulative probability
    CDFd1=ecdf(np.abs(cd1))

    # Compute the slopes for each CDF using (ymax-ymin)/(xmax-xmin) adapted to the CDF 
    slope_a=(CDFa[1][-1]-CDFa[1][0])/(CDFa[0][-1]-CDFa[0][0])
    slope_d1=(CDFd1[1][-1]-CDFd1[1][0])/(CDFd1[0][-1]-CDFd1[0][0])

    return slope_a,slope_d1 

def coeffs_wavelet(y,wavelet_name,level,mode):
    '''
    Inputs:
    - y            <-- signal on which the DWD is applied
    - wavelet name <-- name of the wavelet family to use
    - level        <-- number of detail coefficients
    - mode         <-- mode to use for DWt (str)

    Outputs:
    - ca           <-- approximation coefficient array
    - cd1          <-- first detail coefficient array
    '''

    coeffs=pywt.wavedec(y,wavelet_name,mode,level) # discrete wavelet decomposition from the pywt library, it returns the coefficients 
    ca=coeffs[0]                                   # such as the approximation one is the first of the list and the first detail one the last
    cd1=coeffs[len(coeffs)-1]                 

    return ca,cd1


#-----------------------------------------------------------------------#
#--------------------- mask of artefacts detection ---------------------#
#-----------------------------------------------------------------------#

def max_value_detection_mask(y, fs, M, max_gap):
    '''
    Consider all values above M to be artefact. 
    If detected segments are separated by less than max_gap, everything in between is considered an artefact
    '''

    abs_y = np.abs(y)
    mask_max_value = np.zeros_like(y)
    mask_max_value[np.where(abs_y >= M)[0]] = 1

    # translate the max_gap from time duration to interval length
    max_gap = int(fs * max_gap)
    h = np.ones(max_gap)

    # routine to erode, dilate and erode the mask
    new_mask = sc.ndimage.binary_closing(mask_max_value, structure=h).astype(int)

    # get logical or with new array so possible 1 that are remove are kept since they are in the original array
    mask_max_value = np.logical_or(mask_max_value, new_mask).astype(int)

    return mask_max_value

def correct_peaks_mask(y, fs, f_cut, threshold):

    N = len(y)

    lf_signal = filter_butterworth(y,fs,[0.1,f_cut])

    # Find peaks and compute their properties
    peaks, properties = sc.signal.find_peaks(lf_signal, 
                                          width=[8,32]
                                          )
    
    prominences = properties['prominences']
    widths = properties['widths']
    
    # fixed threshold value
    prominence_threshold = threshold 
    
    # Detect peaks that satisfy the conditions
    relevant_peaks = []
    widths_peaks = []
    for i, peak in enumerate(peaks):
        if prominences[i] > prominence_threshold:
            relevant_peaks.append(peak)
            widths_peaks.append(widths[i])

    mask_peaks=np.zeros(N)
    for i in range(len(relevant_peaks)):
        start = int(relevant_peaks[i] - widths_peaks[i])
        end = int(relevant_peaks[i] + widths_peaks[i])

        if start < 0:
            start = 0
        if end >= N:
            end = N - 1
    
        mask_peaks[start:end]=1
    
    return mask_peaks

def correct_peaks_mask_2(y, fs, threshold):

    N = len(y)

    signal = sc.signal.savgol_filter(y,int(fs/16),1)

    # Find peaks and compute their properties
    peaks, properties = sc.signal.find_peaks(signal, 
                                          width=[8,32],
                                          rel_height=0.5,
                                          wlen= int(fs/4)
                                          )
    
    prominences = properties['prominences']
    widths = properties['widths']
    
    # fixed threshold value
    prominence_threshold = threshold 
    
    # Detect peaks that satisfy the conditions
    relevant_peaks = []
    widths_peaks = []
    for i, peak in enumerate(peaks):
        if prominences[i] > prominence_threshold:
            relevant_peaks.append(peak)
            widths_peaks.append(widths[i])

    mask_peaks=np.zeros(N)
    for i in range(len(relevant_peaks)):
        start = int(relevant_peaks[i] - widths_peaks[i])
        end = int(relevant_peaks[i] + widths_peaks[i])

        if start < 0:
            start = 0
        if end >= N:
            end = N - 1
    
        mask_peaks[start:end]=1
    
    return mask_peaks

def correct_peaks_mask_3(y, fs, threshold):

    N = len(y)

    low_smoothing = sc.signal.savgol_filter(y,int(fs/16),1)
    high_smoothing = sc.signal.savgol_filter(y,int(fs/2),1)

    signal = low_smoothing - high_smoothing

    # Find peaks and compute their properties
    peaks, properties = sc.signal.find_peaks(signal, 
                                          width=[8,32],
                                          rel_height=0.5,
                                          wlen= int(fs/4) 
                                          )
    
    prominences = properties['prominences']
    widths = properties['widths']
    
    # fixed threshold value
    prominence_threshold = threshold 
    
    # Detect peaks that satisfy the conditions
    relevant_peaks = []
    widths_peaks = []
    for i, peak in enumerate(peaks):
        if prominences[i] > prominence_threshold:
            relevant_peaks.append(peak)
            widths_peaks.append(widths[i])

    mask_peaks=np.zeros(N)
    for i in range(len(relevant_peaks)):
        start = int(relevant_peaks[i] - widths_peaks[i])
        end = int(relevant_peaks[i] + widths_peaks[i])

        if start < 0:
            start = 0
        if end >= N:
            end = N - 1
    
        mask_peaks[start:end]=1
    
    return mask_peaks, signal

def grad_mask_artefact(y, fs, division_factor=16, T=5, q=0.9, threshold_max=50):

    # smooth signal
    y_smooth = sc.signal.savgol_filter(y, int(fs / division_factor), 1)
    # get the gradient
    abs_grad_y_smooth = np.abs(np.gradient(y_smooth))
    # threshold for artefacts
    threshold = np.quantile(abs_grad_y_smooth, q) * T
    # get mask of artefacts
    mask = (abs_grad_y_smooth > threshold) | (abs_grad_y_smooth > threshold_max)

    return mask.astype(int)

def dwt_artifacts_mask(y,
                   Ws,
                   step,
                   threshold,
                   wavelet_name,
                   level,
                   mode):
    '''
    Inputs:
    - y              <-- raw eeg signal
    - Ws             <-- sliding window size on wich the DWD is applied
    - step           <-- step size between windows
    - threshold      <-- threshold for the slopes (list of 2, one for the approximation coefficient slope and the other for the first detail coefficient slope)
    - wavelet name   <-- name of the wavelet family to use
    - level          <-- number of detail coefficients
    - mode           <-- mode to use for DWT
    
    Outputs:
    - mask_artifacts <-- 1 at the position where there is no artifacts and 0 where there is one
    - index_mask     <-- marks the position of the iteration each time there is a change in 1/0
    '''

    N = len(y)      # length of the EEG signal that is sent as input (should be the same as the batch of data gathered from the device before doing compution an ideally should be a multiple of Ws)

    # gathering the slopes, thresholding and creating mask
    start_window = 0                              # marker of the position of the left window edge for the next computation

    # creating the mask    
    threshold_a, threshold_d1 = threshold          # receives the thresholds value
    mask_artifacts = [1 for i in range(N)]        # initialize the mask
    mask_pos=[]                                 # set iteration indexes of where the artefacts start and end. (useful for Clean_EEG.py to avoid an unecessary search index in a list to get positions of 0 in the mask) 
    
    # store time and coeeficients list
    t_list = []
    s_a_list = []
    s_d1_list = []

    while start_window<=N:
        if start_window+Ws<=N:                           # checking that the marker for next computation + the size of the window is within the length of the signal

            s_a,s_d1 = CDF_Slope(y[start_window:start_window+Ws],wavelet_name,level,mode)

            t_list.append(start_window+Ws)
            s_a_list.append(s_a)
            s_d1_list.append(s_d1)
        
            if (s_a<threshold_a or s_d1<threshold_d1):# or med_amp>=90) :  # slope lower than threshold_a indicating an artifact EOG or Motion and lower than threshold_d1 indicating an artifact EMG
                mask_artifacts[start_window:start_window+Ws] = [0 for i in range(Ws)]
                mask_pos.append([start_window,start_window+Ws]) # in case
            start_window += step                           # will go above n in case of equality and stops the cycle

        elif (start_window < N and start_window + Ws > N):     # still within the signal but next window will have a part in the signal and another outside. si on prend plus de signal (au moin step en iteration de plus) ne pose plus de pblm (voir avec pour le spectro)
            
            s_a,s_d1 = CDF_Slope(y[start_window:],wavelet_name,level,mode)

            t_list.append(start_window+Ws)
            s_a_list.append(s_a)
            s_d1_list.append(s_d1)

            if (s_a<threshold_a or s_d1<threshold_d1): # or med_amp>=90):    
                mask_artifacts[start_window:] = [0 for i in range(N-start_window)]
                mask_pos.append([start_window,N-1]) # in case
            start_window += Ws                             # will be higher than N and this stops the cycle

    return mask_artifacts, t_list, s_a_list, s_d1_list

def peaks_baseline_mask(y, fs):

    N = len(y)
    x = np.arange(N)
    baseline_fitter = Baseline(x_data = x)

    bkg = baseline_fitter.mor(y, half_window = int(fs/4))[0]

    return bkg

def f_int_artefacts(y, fs, f_int, delta_t_smoothing, T, q):
    '''
    get mask of artefacts within f_int
    '''

    n = int(delta_t_smoothing * fs)
    y2_part = np.convolve(filter_butterworth(y, fs, f_int)**2, np.ones(n)/n, mode = 'same')
    mask = np.zeros_like(y2_part)
    mask[np.where(y2_part > T * np.quantile(y2_part, q))[0]] = 1

    # dilation in case not enough of the artefact is detected
    h = np.ones(n)
    mask = sc.ndimage.binary_dilation(mask, structure=h).astype(int)

    return mask

def f_int_artefacts_sc(y, fs, f_int, delta_t_smoothing, T, q):
    '''
    get mask of artefacts within f_int
    '''

    N_smoothing = int(delta_t_smoothing * fs)
    y2_part = sc.signal.savgol_filter(filter_butterworth(y, fs, f_int)**2, int(delta_t_smoothing * fs), 1)
    mask = np.zeros_like(y2_part)
    mask[np.where(y2_part > T * np.quantile(y2_part, q))[0]] = 1

    # dilation in case not enough of the artefact is detected
    h = np.ones(N_smoothing)
    mask = sc.ndimage.binary_dilation(mask, structure=h).astype(int)

    return mask

def detect_artefact_env_prominence(y, fs, f_int, prominence_threshold = 10, width_division_factor = [16,4], wlen_division_factor = 4):

    # filter into frequency interval of interest
    signal = filter_butterworth(y, fs, f_int)
    # get envelope of signal
    env = diff_envelops(signal)
    #env = env / (np.quantile(env,0.9) + 1)
    # Find peaks and compute their properties
    peaks, properties = sc.signal.find_peaks(env, 
                                            prominence = prominence_threshold,
                                            width = [int(fs / width_division_factor[0]), int(fs / width_division_factor[-1])],
                                            rel_height = 0.5,
                                            wlen = int(fs / wlen_division_factor) 
                                          )
    widths = properties['widths']

    # mask of artefacts
    N = len(env)
    mask = np.zeros(N)
    for i in range(len(peaks)):
        start = int(peaks[i] - widths[i])
        end = int(peaks[i] + widths[i])

        if start < 0:
            start = 0
        if end >= N:
            end = N - 1
    
        mask[start:end] = 1

    return mask, signal, env, peaks, widths, properties

def detect_artefact_env_prominence_quantile(y, fs, f_int, delta_t_smoothing, width_division_factor = [16,4], wlen_division_factor = 4, T = 50, q = 0.5):

    # filter into frequency interval of interest
    signal = filter_butterworth(y, fs, f_int)
    # get smooth power of signal
    y2_part = sc.signal.savgol_filter(signal**2, int(delta_t_smoothing * fs), 1)

    # Find peaks and compute their properties
    peaks, properties = sc.signal.find_peaks(y2_part, 
                                            width = [int(fs / width_division_factor[0]), int(fs / width_division_factor[-1])],
                                            rel_height = 0.5,
                                            wlen = int(fs / wlen_division_factor) 
                                          )
    widths = properties['widths']
    prominences = properties['prominences']

    # get quantile value of prominence
    val = np.quantile(prominences, q) * T

    # mask of artefacts
    N = len(y2_part)
    mask = np.zeros(N)
    for i in range(len(peaks)):
        if prominences[i] >= val:# or prominences[i] >= 400:
            start = int(peaks[i] - widths[i])
            end = int(peaks[i] + widths[i])

            if start < 0:
                start = 0
            if end >= N:
                end = N - 1
        
            mask[start:end] = 1

    return mask

def detect_artefact_env_prominence_z_score(y, fs, f_int, width_division_factor = [16,4], wlen_division_factor = 4):

    # filter into frequency interval of interest
    signal = filter_butterworth(y, fs, f_int)
    # get envelope of signal
    env = diff_envelops(signal)
    #env = env / (np.quantile(env,0.9) + 1)
    # Find peaks and compute their properties
    peaks, properties = sc.signal.find_peaks(env, 
                                            width = [int(fs / width_division_factor[0]), int(fs / width_division_factor[-1])],
                                            rel_height = 0.5,
                                            wlen = int(fs / wlen_division_factor) 
                                          )
    widths = properties['widths']
    prominences = properties['prominences']

    # get outliers mask with z-score
    mask_outliers = is_outlier_zscore(prominences)

    # mask of artefacts
    N = len(env)
    mask = np.zeros(N)
    for i in range(len(peaks)):
        #if env[peaks[i]] >= 3 * val and prominences[i] >= val:
        if mask_outliers[i] == 1:
            start = int(peaks[i] - widths[i])
            end = int(peaks[i] + widths[i])

            if start < 0:
                start = 0
            if end >= N:
                end = N - 1
        
            mask[start:end] = 1

    return mask, signal, env, peaks, widths, properties

def detect_artefact_env_prominence_iqr(y, fs, f_int, width_division_factor = [16,4], wlen_division_factor = 4, env_diff = False):

    # filter into frequency interval of interest
    signal = filter_butterworth(y, fs, f_int)
    # get envelope of signal
    n = int(fs/16)
    env = np.convolve(signal**2, np.ones(n)/n, mode='same') #diff_envelops(signal)
    #env = env / (np.quantile(env,0.9) + 1)
    # Find peaks and compute their properties
    peaks, properties = sc.signal.find_peaks(env, 
                                            width = [0],
                                            rel_height = 0.5,
                                            wlen = int(fs / wlen_division_factor) 
                                          )
    widths = properties['widths']
    prominences = properties['prominences']

    # mask of artefacts
    N = len(env)
    mask = np.zeros(N)
    if len(prominences) >= 1:
        # get outliers mask with z-score
        mask_outliers, val = is_outlier_iqr(prominences)
        for i in range(len(peaks)):
            #if env[peaks[i]] >= 3 * val and prominences[i] >= val:
            if mask_outliers[i] == 1:#or env[peaks[i]] >= 1.5 * val:
                start = int(peaks[i] - widths[i])
                end = int(peaks[i] + widths[i])

                if start < 0:
                    start = 0
                if end >= N:
                    end = N - 1
            
                mask[start:end] = 1

    else:
        val = 0

    return mask, signal, env, peaks, widths, properties, val

# def detect_artefact_env_lof(y, fs, f_int, diff_env = False):

#     # filter into frequency interval of interest
#     signal = filter_butterworth(y, fs, f_int)

#     # T_upper = np.quantile(signal, 0.9) * 10
#     # T_lower = np.quantile(signal, 0.1) / 10
#     # signal[signal >= T_upper] = T_upper
#     # signal[signal<= T_lower] = T_lower
#     # get envelope of signal
#     n = int(fs/16)
#     env = np.convolve(signal**2, np.ones(n)/n, mode='same') 
#     N = len(signal)
#     t = np.linspace(0, N/fs, N)
#     # downsample
#     env = env[::4]
#     t = t[::4]
#     if diff_env:
#         env = env - sc.signal.savgol_filter(env,int(fs/4/5),1)
    
#     n = int(3*fs/4)

#     # Prepare data for LOF (LOF expects 2D array)
#     X = np.column_stack((t, env))

#     # Apply LOF
#     lof = LocalOutlierFactor(n_neighbors = n)
#     y_pred = lof.fit_predict(X)
#     lof_scores = -lof.negative_outlier_factor_

#     # Identify outliers
#     mask = (lof_scores >= 3).astype(int)

#     return mask, lof_scores

def get_mask_artefacts(y, fs,
                      edge_frequency, delta_t_edge_frequency,       # mask burst
                      f_int_hf, delta_t_smoothing_hf, T_hf, q_hf,   # mask hf
                      T_peaks,                                      # mask peaks in signal
                      ):
    
    N = len(y)

    #--- get mask burst
    mask_burst = get_burst(edge_frequency, delta_t_edge_frequency)
    mask_burst_resized = resize_binary_mask(mask_burst, N)

    #--- get mask hf
    mask_hf = f_int_artefacts(y, fs, f_int_hf, delta_t_smoothing_hf, T_hf, q_hf)

    #--- get mask peaks
    mask_peaks_2 = correct_peaks_mask_2(y, fs, T_peaks)
    mask_peaks_3 = correct_peaks_mask_3(y, fs, T_peaks)[0]
    mask_peaks = np.logical_or(mask_peaks_2, mask_peaks_3).astype(int)

    #--- delete artefacts within burst
    mask_hf = np.logical_and(mask_hf == 1, mask_burst_resized == 0).astype(int)
    mask_peaks = np.logical_and(mask_peaks == 1, mask_burst_resized == 0).astype(int)

    #--- combine masks
    mask_artefacts = np.logical_or(mask_peaks, mask_hf).astype(int)

    return mask_artefacts

def get_mask_artefacts_full(y, fs,
                      edge_frequency, delta_t_edge_frequency,       # mask burst
                      f_int_hf, delta_t_smoothing_hf, T_hf, q_hf,   # mask hf
                      T_peaks,                                      # mask peaks in signal
                      ):
    
    N = len(y)

    #--- get mask burst
    mask_burst = get_burst(edge_frequency, delta_t_edge_frequency)
    mask_burst_resized = resize_binary_mask(mask_burst, N)

    #--- get mask hf
    mask_hf = f_int_artefacts(y, fs, f_int_hf, delta_t_smoothing_hf, T_hf, q_hf)

    #--- get mask peaks
    mask_peaks_2 = correct_peaks_mask_2(y, fs, T_peaks)
    mask_peaks_3 = correct_peaks_mask_3(y, fs, T_peaks)[0]
    mask_peaks = np.logical_or(mask_peaks_2, mask_peaks_3).astype(int)

    #--- delete artefacts within burst
    mask_hf = np.logical_and(mask_hf == 1, mask_burst_resized == 0).astype(int)
    mask_peaks = np.logical_and(mask_peaks == 1, mask_burst_resized == 0).astype(int)

    #--- combine masks
    mask_artefacts = np.logical_or(mask_peaks, mask_hf).astype(int)

    return mask_artefacts, mask_burst_resized, mask_hf, mask_peaks

#---------------------------------------------------------------------------------------------#
#                            Artifacts detection through sliding windows                      #
#---------------------------------------------------------------------------------------------#

def sliding_f_int_artifacts(y, fs, f_int, Ws, step, T_threshold, q_threshold, q, mode = 'quantile'):

    #--- filter into frequency range of interest
    signal = filter_butterworth(y, fs, f_int)

    #--- get sliding windows
    windows = [signal[i:i+Ws]**2 for i in range(0,len(y)-Ws,step)]

    #--- get power of signal per sliding window
    if mode == 'quantile':
        P_list = np.array([np.quantile(win, q) for win in windows])
    elif mode == 'mean':
        P_list = np.array([np.mean(win) for win in windows])

    #--- resize to same length as signal for easy manipulation of mask
    M = len(signal)
    N = len(P_list)
    indices = np.linspace(0, N-1, num=M)
    indices_rounded = np.round(indices).astype(int)
    P_list = P_list[indices_rounded]    

    #--- threshold
    threshold = T_threshold * np.quantile(P_list, q_threshold)

    #--- mask 
    mask = np.zeros_like(signal)
    mask[P_list >= threshold] = 1

    return mask, P_list
    
    

#----------------------------------------- sliding windows -----------------------------------#

# threshold on power
def sliding_adaptive_thresholding(signal, T_full_signal, q_full_signal, Ws, step, T_window, q_window):

    #--- full signal adaptive thresholding
    N_signal = len(signal)
    print(N_signal)
    mask_full_signal = np.zeros(N_signal)
    threshold_full_signal = T_full_signal * np.quantile(signal, q_full_signal)
    mask_full_signal[np.where(signal > threshold_full_signal)[0]] = 1

    list_mask_window = []
    for i in range(0,len(signal)-Ws+1,step):
        #--- create mask for current window
        window = signal[i:i+Ws]
        threshold_window = T_window * np.quantile(window, q_window)
        mask_window = np.zeros_like(window)
        mask_window[np.where(window > threshold_window)[0]] = 1
        #--- set mask in format for the full signal
        current_mask = np.zeros(N_signal)
        current_mask[i:i+Ws] = mask_window 
        list_mask_window.append(current_mask)

    return mask_full_signal, np.array(list_mask_window)



# thresholod on prominence
def get_peaks_and_prominences_mask(x, prominence_min, Wlen, prominence_factor):
    """
    Compute peaks and their prominences, keeping only the most prominent ones.

    Parameters
    ----------
    x : 1D array-like
        Signal values.
    prominence_min : float or None
        Minimum prominence for peak detection (passed to scipy.find_peaks).
    prominence_factor : float
        Keep only peaks whose prominence >= factor * median(prominences).

    Returns
    -------
    peaks : np.ndarray
        Indices of filtered peaks.
    prominences : np.ndarray
        Prominence values of filtered peaks.
    left_bases : np.ndarray
        Indices of left bases for filtered peaks.
    right_bases : np.ndarray
        Indices of right bases for filtered peaks.
    """
    # Step 1: detect peaks (using any minimal prominence threshold)
    peaks, _ = find_peaks(x, prominence=prominence_min)
    prominences, left_bases, right_bases = peak_prominences(x, peaks, wlen = Wlen)
    mask_artifacts = np.zeros_like(x)

    # Step 2: compute the median prominence
    if len(prominences) > 0:
        median_prom = np.median(prominences)
        threshold = prominence_factor * median_prom
        # Step 3: keep only peaks above threshold
        mask = prominences >= threshold
        peaks, prominences = peaks[mask], prominences[mask]
        left_bases, right_bases = left_bases[mask], right_bases[mask]
        for i in range(len(left_bases)):
            mask_artifacts[left_bases[i] : right_bases[i]] = 1
    
    return mask_artifacts

def sliding_prominence_adaptive_thresholding(signal, Ws, step, prominence_min=None, Wlen = 64, prominence_factor=10.0):

    #--- full signal mask
    N_signal = len(signal)
    print(N_signal)
    mask_full_signal = get_peaks_and_prominences_mask(signal, prominence_min, Wlen, prominence_factor)

    list_mask_window = []
    for i in range(0,len(signal)-Ws+1,step):
        #--- create mask for current window
        window = signal[i:i+Ws]
        current_mask_prominence = get_peaks_and_prominences_mask(window, prominence_min, Wlen, prominence_factor)
        #--- set mask in format for the full signal
        current_mask = np.zeros(N_signal)
        current_mask[i:i+Ws] = current_mask_prominence
        list_mask_window.append(current_mask)

    return mask_full_signal, np.array(list_mask_window)