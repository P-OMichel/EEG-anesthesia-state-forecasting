import numpy as np
import scipy as sc
from Functions.filter import get_filtered_signal, filter_butterworth
from Functions.detect_artifacts import find_artifacts, get_artifacts_and_gc
from Functions.WaveletQuantileNormalization import WQN_3
from Functions.main_freq import get_main_frequency_zcr
from Functions.utils import get_quantile
from Functions.time_frequency import spectrogram
from Functions.edge_frequency import edge_frequencies_limit_value, edge_frequencies_significant_value
from Functions.suppressions import detect_supp_edge_frequency

class ComputeVariables:

    def __init__(self, fs):
        super().__init__()

        self.fs = fs

        self.list_detection = [2*self.fs, 1*self.fs, [0.003,0.03], "sym4", 4, "periodization"] 
        self.list_WQN = ["sym4", "periodization", 30, 1]
    
    def clean_signal(self, current_signal):
        print(f'shape signal: {np.shape(current_signal)}')

        self.y = current_signal - np.median(current_signal)
        self.y_corr = self.y

        self.prop_gc = 0
        try:
            #--- detect artefacts
            # detection of the artifacts
            self.prop_gc, index_mask = get_artifacts_and_gc(self.y,*self.list_detection)[-2:]      

            # if it has some artifacts  
            if len(index_mask) != 0:                                              
            #--- clean the artifacts using WQN algorithm
                self.y_corr = WQN_3(self.y,index_mask,*self.list_WQN)  

            print('artifacts corrected')
        except:  
            self.y_corr = self.y

        print(f'proportion ground check: {self.prop_gc}')

    
    def get_P_list(self):

        self.signals = get_filtered_signal(self.y_corr, self.fs, [[0.5,4],[7,15],[15,30],[30,45]])
        self.P_list = np.median(self.signals**2, axis = 1)

    def get_P_prop(self):

        P_tot = np.sum(self.P_list)

        self.prop_delta = self.P_list[0] / P_tot
        self.prop_alpha = self.P_list[1] / P_tot
        self.prop_beta = self.P_list[2] / P_tot
        self.prop_gamma = self.P_list[-1] / P_tot

    def get_P_ratios(self):

        self.r_alpha_delta = self.P_list[1] / self.P_list[0]
        self.r_beta_delta = self.P_list[2] / self.P_list[0]
        self.r_gamma_delta = self.P_list[-1] / self.P_list[0]
        self.r_beta_alpha = self.P_list[2] / self.P_list[1]
        self.r_gamma_alpha = self.P_list[-1] / self.P_list[1]
        self.r_gamma_beta = self.P_list[-1] / self.P_list[2]

    def get_f_main(self):

        self.f_central = get_main_frequency_zcr(self.y_corr, self.fs)

    def get_f_quantiles(self):

        #--- compute psd 
        f, Pxx_den = sc.signal.welch(self.y_corr, self.fs, nperseg = self.fs, noverlap = int(0.9*self.fs), nfft = 2 * self.fs, average = 'median')
        # cut at f = 45
        delta_f = f[1]-f[0]
        j = int((45 - f[0])/delta_f)
        f, Pxx_den = f[:j], Pxx_den[:j]
        #--- get quantile values
        self.q_0_85 = get_quantile(Pxx_den, f, 0.85)

    def get_ef(self):

        sqrt_med = np.sqrt(np.median(self.y**2)) # need to be centered around 0 already
        factor = 25 / sqrt_med
        if factor <= 1:
            win_norm = self.y * factor
        else:
            win_norm = self.y

        t_spectro, f_spectro, spectro = spectrogram(win_norm, self.fs, nperseg_factor= 1, nfft_factor=2)
        self.ef = edge_frequencies_limit_value(spectro, f_spectro, factor_hf = 5, threshold_min = 5, threshold_max=12, threshold=None, T=1, q=0.95)[0]
        #self.ef_recovery = edge_frequencies_limit_value(spectro, f_spectro, factor_hf = 5, threshold_min = 0.1, threshold_max=12, threshold=None, T=1, q=0.95)[0]
        self.fs_ef = 1 / (t_spectro[1] - t_spectro[0])

        #self.ef_0_1 = np.quantile(self.ef, 0.1)
        #self.ef_0_1_recovery = np.quantile(self.ef_recovery, 0.1)
        #self.ef_0_5 = np.median(self.ef)
        #self.ef_0_5_recovery = np.median(self.ef_recovery)

    def get_supp(self):

        self.ies_prop, self.alpha_supp_prop = detect_supp_edge_frequency(self.ef, self.fs_ef)[-2:]
        if self.prop_gc >  0:
            self.ies_prop = self.ies_prop - (self.prop_gc + 0.2)
        #self.ies_prop_recovery, self.alpha_supp_prop_recovery = detect_supp_edge_frequency(self.ef_recovery, self.fs_ef)[-2:]
        #if self.prop_gc >  0:
            #self.ies_prop_recovery = self.ies_prop_recovery - (self.prop_gc + 0.2)

class ComputeVariablesReview:

    def __init__(self, fs):
        super().__init__()

        self.fs = fs

        self.list_detection = [2*self.fs, 1*self.fs, [0.003,0.03], "sym4", 4, "periodization"] 
        self.list_WQN = ["sym4", "periodization", 30, 1]
    
    def clean_signal(self, current_signal):
        print(f'shape signal: {np.shape(current_signal)}')

        self.y = current_signal - np.median(current_signal)
        self.y_corr = self.y

        self.prop_gc = 0
        try:
            #--- detect artefacts
            # detection of the artifacts
            self.prop_gc, index_mask = get_artifacts_and_gc(self.y,*self.list_detection)[-2:]      

            # if it has some artifacts  
            if len(index_mask) != 0:                                              
            #--- clean the artifacts using WQN algorithm
                self.y_corr = WQN_3(self.y,index_mask,*self.list_WQN)  

            print('artifacts corrected')
        except:  
            self.y_corr = self.y

        print(f'proportion ground check: {self.prop_gc}')
    
    def get_P_list(self):

        self.signals = get_filtered_signal(self.y_corr, self.fs, [[0.5,4],[7,15],[15,30],[30,45]])
        self.P_list = np.median(self.signals**2, axis = 1)

    def get_P_prop(self):

        P_tot = np.sum(self.P_list)

        self.prop_delta = self.P_list[0] / P_tot
        self.prop_alpha = self.P_list[1] / P_tot
        self.prop_beta = self.P_list[2] / P_tot
        self.prop_gamma = self.P_list[-1] / P_tot

    def get_P_ratios(self):

        self.r_alpha_delta = self.P_list[1] / self.P_list[0]
        self.r_beta_delta = self.P_list[2] / self.P_list[0]
        self.r_gamma_delta = self.P_list[-1] / self.P_list[0]
        self.r_beta_alpha = self.P_list[2] / self.P_list[1]
        self.r_gamma_alpha = self.P_list[-1] / self.P_list[1]
        self.r_gamma_beta = self.P_list[-1] / self.P_list[2]

    def get_f_main(self):

        self.f_central = get_main_frequency_zcr(self.y_corr, self.fs)

    def get_f_quantiles(self):

        #--- compute psd 
        f, Pxx_den = sc.signal.welch(self.y_corr, self.fs, nperseg = self.fs, noverlap = int(0.9*self.fs), nfft = 2 * self.fs, average = 'median')
        # cut at f = 45
        delta_f = f[1]-f[0]
        j = int((45 - f[0])/delta_f)
        f, Pxx_den = f[:j], Pxx_den[:j]
        #--- get quantile values
        self.q_0_85 = get_quantile(Pxx_den, f, 0.85)

    def get_ef(self):

        sqrt_med = np.sqrt(np.median(self.y**2)) # need to be centered around 0 already
        factor = 25 / sqrt_med
        if factor <= 1:
            win_norm = self.y * factor
        else:
            win_norm = self.y

        t_spectro, f_spectro, spectro = spectrogram(win_norm, self.fs, nperseg_factor= 1, nfft_factor=2)
        self.ef = edge_frequencies_limit_value(spectro, f_spectro, factor_hf = 5, threshold_min = 5, threshold_max=12, threshold=None, T=1, q=0.95)[0]
        #self.ef_recovery = edge_frequencies_limit_value(spectro, f_spectro, factor_hf = 5, threshold_min = 0.1, threshold_max=12, threshold=None, T=1, q=0.95)[0]
        self.fs_ef = 1 / (t_spectro[1] - t_spectro[0])

        #self.ef_0_1 = np.quantile(self.ef, 0.1)
        #self.ef_0_1_recovery = np.quantile(self.ef_recovery, 0.1)
        #self.ef_0_5 = np.median(self.ef)
        #self.ef_0_5_recovery = np.median(self.ef_recovery)

    def get_supp(self):

        self.ies_prop, self.alpha_supp_prop = detect_supp_edge_frequency(self.ef, self.fs_ef)[-2:]
        if self.prop_gc >  0:
            self.ies_prop = max(0, self.ies_prop - (self.prop_gc + 0.2)) 
        #self.ies_prop_recovery, self.alpha_supp_prop_recovery = detect_supp_edge_frequency(self.ef_recovery, self.fs_ef)[-2:]
        #if self.prop_gc >  0:
            #self.ies_prop_recovery = self.ies_prop_recovery - (self.prop_gc + 0.2)


