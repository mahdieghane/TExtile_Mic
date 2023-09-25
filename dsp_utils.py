import numpy as np
import scipy
from scipy import signal
from scipy.fft import rfft, hfft
from scipy.stats import skew, kurtosis, hmean, moment
import pywt
import pandas as pd
from tsfresh import extract_features as ts_extract_features
from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsfeature1 import calculate_tsfresh_features
import time




class DSPUtils:
    SAMPLE_FREQUENCY = 1024

    ### calculate FFT of the signal with the fixed window size (padding 0 if the length of signal is less than the window size)
    @staticmethod
    def calculate_fft(sig, window_size):
        fft = rfft(sig, window_size)
        return np.abs(fft)

    ### downsample the signal with the sample ratio (0~1)
    @staticmethod
    def downsample(sig, ratio):
        new_signal = []
        for i in range(int(len(sig)*ratio)):
            new_signal.append(np.mean(sig[int(i/ratio):int((i+1)/ratio)]))
        return np.array(new_signal)

    ### apply overlapping sliding windows on the signal
    @staticmethod
    def apply_sliding_window(sig, window_size, shift_size):

        shift = shift_size
        windows = []
        index = 0
        while index < len(sig):
            end = min(index + window_size, len(sig))
            windows.append((index, sig[index:end]))
            index += int(shift)
        return windows


    ### apply 500Hz low pass filter on the signals
    @staticmethod
    def apply_low_pass_filter(sig):
        sos = scipy.signal.butter(4, 500, 'lowpass', fs=DSPUtils.SAMPLE_FREQUENCY, output='sos')
        y_sos = scipy.signal.sosfilt(sos, sig)
        return sig

    ### apply bandstop filters to remove power line noise
    @staticmethod
    def remove_power_line_noise(sig):
        sos = scipy.signal.butter(4, [55, 65], 'bandstop', fs=DSPUtils.SAMPLE_FREQUENCY, output='sos')
        y_sos = scipy.signal.sosfilt(sos, sig)
        sos2 = scipy.signal.butter(4, [115, 125], 'bandstop', fs=DSPUtils.SAMPLE_FREQUENCY, output='sos')
        y_sos2 = scipy.signal.sosfilt(sos2, y_sos)
        sos3 = scipy.signal.butter(4, [295, 305], 'bandstop', fs=DSPUtils.SAMPLE_FREQUENCY, output='sos')
        y_sos3 = scipy.signal.sosfilt(sos, y_sos2)
        return sig


    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
    @staticmethod
    def apply_window_filter(sig ,window='blackmanharris'):

        win = scipy.signal.get_window(window, FILTER_WINDOW_SIZE)

        filtered_signal= scipy.signal.convolve(sig, win, mode='same') / sum(win)

        return sig

    @staticmethod
    def calculate_fft_energy_along_windows(fft_windows):
        energy = []
        for fft_window in fft_windows:
            energy.append(np.sum(np.abs(fft_window)))
        return energy

    @staticmethod
    def calculate_maxbin_along_windows(fft_windows):
        time_max_bin = []
        for fft_window in fft_windows:
            time_max_bin.append(np.argsort(fft_window)[-1])
        return time_max_bin
    
    @staticmethod
    def calculate_total_fft_energy(fft_windows):
        energy = 0
        for fft_window in fft_windows:
            energy += np.sqrt(np.mean(np.square(fft_window)))
        return energy

    @staticmethod
    def is_noisy(sig, fft_windows):

        energy = DSPUtils.calculate_total_fft_energy(fft_windows)
        COARSE_ENERGY_THRESHOULD = 0.8 ## this value should be adaptive

        if energy > COARSE_ENERGY_THRESHOULD:
            return False

        return True

    ### calculate the FFT along windows
    @staticmethod
    def convert_windows_to_fft_windows(windows, window_size):
        fft_windows = []

        for sig in windows:
            fft_window = DSPUtils.calculate_fft(sig, window_size)
            fft_windows.append(fft_window)
        return np.array(fft_windows)

     ### low-pass filter the signal along the windows 
    @staticmethod
    def lowpass_filter_along_windows(windows):
        filtered_windows = []

        for sig in windows:

            filtered_signal = sig
            filtered_signal = filtered_signal
            # filtered_signal = apply_window_filter(filtered_signal)
            filtered_windows.append(filtered_signal)
        return np.array(filtered_windows)

    ### substrate background_noise_profile from each fft window
    @staticmethod
    def remove_background_noise_along_fft_windows(fft_windows, background_fft_profile):
        for fft_window in fft_windows:
            fft_window = fft_window- background_fft_profile
            fft_window[fft_window<0] = 0
        return fft_windows

    ### segment the signal along windws
    ### return (concatenated segmented signal, the segmented fft windows)
    @staticmethod
    def segment_along_windows(windows, windows1,windows2,windows3,windows4, window_size, shift_size):
        FINE_ENERGY_THRESHOULD = 0.001 ## should be adaptive
        ENERGY_WINDOWN_SIZE = 150 ## should be adaptive
        
        #filtered_windows = DSPUtils.lowpass_filter_along_windows(windows)
        #print(len(filtered_windows))

        fft_windows = DSPUtils.convert_windows_to_fft_windows(windows, window_size)
        fft_windows1 = DSPUtils.convert_windows_to_fft_windows(windows1, window_size)
        fft_windows2 = DSPUtils.convert_windows_to_fft_windows(windows2, window_size)
        fft_windows3 = DSPUtils.convert_windows_to_fft_windows(windows3, window_size)
        fft_windows4 = DSPUtils.convert_windows_to_fft_windows(windows4, window_size)
        #print(len(fft_windows))        
        
        #fft_windows = DSPUtils.remove_background_noise_along_fft_windows(fft_windows, background_fft_profile)
        #print(len(fft_windows))
        segmented_signal = []
        segmented_signal1 = []
        segmented_signal2 = []
        segmented_signal3 = []
        segmented_signal4 = []
        

        segmented_fft_window = []
        segmented_fft_window1 = []
        segmented_fft_window2 = []
        segmented_fft_window3 = []
        segmented_fft_window4 = []

        segmented_zero_fft_window = []
        segmented_zero_fft_window1 = []
        segmented_zero_fft_window2 = []
        segmented_zero_fft_window3 = []
        segmented_zero_fft_window4 = []

        sliding_window_in_energy = []
        sliding_window_in_energy1 = []
        sliding_window_in_energy2 = []
        sliding_window_in_energy3 = []
        sliding_window_in_energy4 = []


        for index in range(min(len(windows),len(windows1),len(windows2),len(windows3),len(windows4))):
            fft_window = fft_windows[index]
            filtered_signal = windows[index]
            total_fft_energy = np.sqrt(np.mean(np.square(fft_window)))   
            sliding_window_in_energy.append(total_fft_energy)
            if len(sliding_window_in_energy) >  ENERGY_WINDOWN_SIZE:
                sliding_window_in_energy = sliding_window_in_energy[-ENERGY_WINDOWN_SIZE:]
            average_energy = sum(sliding_window_in_energy) / len(sliding_window_in_energy)

            fft_window1 = fft_windows1[index]
            filtered_signal1 = windows1[index]
            total_fft_energy1 = np.sqrt(np.mean(np.square(fft_window1)))   
            sliding_window_in_energy1.append(total_fft_energy1)
            if len(sliding_window_in_energy1) >  ENERGY_WINDOWN_SIZE:
                sliding_window_in_energy1 = sliding_window_in_energy1[-ENERGY_WINDOWN_SIZE:]
            average_energy1 = sum(sliding_window_in_energy1) / len(sliding_window_in_energy1)

            fft_window2 = fft_windows2[index]
            filtered_signal2 = windows2[index]
            total_fft_energy2 = np.sqrt(np.mean(np.square(fft_window2)))   
            sliding_window_in_energy2.append(total_fft_energy2)
            if len(sliding_window_in_energy2) >  ENERGY_WINDOWN_SIZE:
                sliding_window_in_energy2 = sliding_window_in_energy2[-ENERGY_WINDOWN_SIZE:]
            average_energy2 = sum(sliding_window_in_energy2) / len(sliding_window_in_energy2)

            fft_window3 = fft_windows3[index]
            filtered_signal3 = windows3[index]
            total_fft_energy3 = np.sqrt(np.mean(np.square(fft_window3)))   
            sliding_window_in_energy3.append(total_fft_energy3)
            if len(sliding_window_in_energy3) >  ENERGY_WINDOWN_SIZE:
                sliding_window_in_energy3 = sliding_window_in_energy3[-ENERGY_WINDOWN_SIZE:]
            average_energy3 = sum(sliding_window_in_energy3) / len(sliding_window_in_energy3)

            fft_window4 = fft_windows4[index]
            filtered_signal4 = windows[index]
            total_fft_energy4 = np.sqrt(np.mean(np.square(fft_window4)))   
            sliding_window_in_energy4.append(total_fft_energy4)
            if len(sliding_window_in_energy4) >  ENERGY_WINDOWN_SIZE:
                sliding_window_in_energy4 = sliding_window_in_energy4[-ENERGY_WINDOWN_SIZE:]
            average_energy4 = sum(sliding_window_in_energy4) / len(sliding_window_in_energy4)


            if  (average_energy +average_energy1 +average_energy2 +average_energy3 +average_energy4)  > FINE_ENERGY_THRESHOULD:
                """"
                if len(segmented_signal) == 0:
                    segmented_signal = filtered_signal
                    segmented_signal1 = filtered_signal1
                    segmented_signal2 = filtered_signal2
                    segmented_signal3 = filtered_signal3
                    segmented_signal4 = filtered_signal4

                else:
                    segmented_signal =np.concatenate((segmented_signal, filtered_signal[-shift_size:]))
                    segmented_signal1 =np.concatenate((segmented_signal1, filtered_signal1[-shift_size:]))
                    segmented_signal2 =np.concatenate((segmented_signal2, filtered_signal2[-shift_size:]))
                    segmented_signal3 =np.concatenate((segmented_signal3, filtered_signal3[-shift_size:]))
                    segmented_signal4 =np.concatenate((segmented_signal4, filtered_signal4[-shift_size:]))
                """
                segmented_fft_window.append(fft_window.tolist())
                segmented_fft_window1.append(fft_window1.tolist())
                segmented_fft_window2.append(fft_window2.tolist())
                segmented_fft_window3.append(fft_window3.tolist())
                segmented_fft_window4.append(fft_window4.tolist())

            else:
                segmented_zero_fft_window.append(fft_window.tolist())
                segmented_zero_fft_window1.append(fft_window1.tolist())
                segmented_zero_fft_window2.append(fft_window2.tolist())
                segmented_zero_fft_window3.append(fft_window3.tolist())
                segmented_zero_fft_window4.append(fft_window4.tolist())
        # print(segmented_signal) 
        return (segmented_fft_window,
                segmented_fft_window1,
                segmented_fft_window2,
                segmented_fft_window3,
                segmented_fft_window4) 

        #return np.array(fft_windows)


        """"
        segmented_signal = []
        segmented_fft_window = []
        sliding_window_in_energy = []


        for index in range(len(filtered_windows)):
            fft_window = fft_windows[index]
            filtered_signal = filtered_windows[index]

            total_fft_energy = np.sqrt(np.mean(np.square(fft_window)))   
            sliding_window_in_energy.append(total_fft_energy)
            if len(sliding_window_in_energy) >  ENERGY_WINDOWN_SIZE:
                sliding_window_in_energy = sliding_window_in_energy[-ENERGY_WINDOWN_SIZE:]

            average_energy = sum(sliding_window_in_energy) / len(sliding_window_in_energy)
            if  average_energy > FINE_ENERGY_THRESHOULD:
                if len(segmented_signal) == 0:
                    segmented_signal = filtered_signal
                else:
                    segmented_signal = np.concatenate((segmented_signal, filtered_signal[-shift_size:]))

                segmented_fft_window.append(fft_window.tolist())

            elif len(segmented_signal) > 0:
                break
        print(segmented_signal) 
        """

    ### calculate continuous wavelet trasform coefficients (too slow to be real time on my mac)
    @staticmethod
    def calculate_cwt_coefficient(sig, window_size):
        widths = np.arange(window_size/16, window_size/2, window_size/16)
        # print(sig.shape)
        smaller_sig = signal.resample(sig, int(window_size))
        cwtmatr, frequencies = pywt.cwt(smaller_sig, widths, 'mexh')
        # print(cwtmatr.shape)
        # print(frequencies)
        return cwtmatr

    ### calculate stats of data, including max, mean, min, std and kurtosis
    @staticmethod
    def stats_describe(data):
        result = [np.max(data), np.mean(data), np.min(data), np.std(data),  kurtosis(data)]
        return result


    @staticmethod
    def extract_feature( fft_windows,fft_windows1,fft_windows2,fft_windows3,fft_windows4 ):

        features = np.array([])
        ### calculate time-domain features (not used)
        ### calculate frequency-domain features
        if fft_windows:
            np_fft = np.array(fft_windows)
            fft_len = [len(fft_windows)]
            fft_mean = np.mean(np_fft, axis = 0)
            fft_quantile1st = np.quantile(np_fft, 0.25, axis = 0)
            fft_quantile3th = np.quantile(np_fft, 0.75, axis = 0)
            fft_median = np.median(np_fft, axis = 0)
            fft_max = np.amax(np_fft, axis = 0)
            fft_std = np.std(np_fft, axis = 0) #asymmetric
            fft_hmean =hmean(np_fft, axis = 0) #asymmetric
            fft_moment = moment(np_fft, axis = 0) 
            #fft_skew = skew(np_fft, axis = 0) #asymmetric
            #fft_kurtosis= kurtosis(np_fft, axis = 0) #shape
            #fft_time_skew = DSPUtils.stats_describe(skew(np_fft, axis = 1)) #asymmetric
            #fft_time_kurtosis= DSPUtils.stats_describe(kurtosis(np_fft, axis = 1)) #shape
            fft_time_max = DSPUtils.stats_describe(np.max(np_fft, axis = 1))
            fft_time_mean = DSPUtils.stats_describe(np.mean(np_fft, axis = 1))
            fft_time_std = DSPUtils.stats_describe(np.std(np_fft, axis = 1))
            magnitude = np.abs(np_fft)
            frequency_bins = np.fft.fftfreq(len(np_fft))
            spectral_energy = np.sum(magnitude**2)
            #dominant_frequency = np.abs(frequency_bins[np.argmax(magnitude)])
            #spectral_entropy = -np.sum(magnitude**2 * np.log(magnitude**2))



            np_fft1 = np.array(fft_windows1)
            fft_len1 = [len(fft_windows1)]
            fft_mean1 = np.mean(np_fft1, axis = 0)
            fft_quantile1st1 = np.quantile(np_fft1, 0.25, axis = 0)
            fft_quantile3th1 = np.quantile(np_fft1, 0.75, axis = 0)
            fft_median1 = np.median(np_fft1, axis = 0)
            fft_max1 = np.amax(np_fft1, axis = 0)
            fft_std1 = np.std(np_fft1, axis = 0) #asymmetric
            fft_hmean1 =hmean(np_fft1, axis = 0) #asymmetric
            fft_moment1 = moment(np_fft1, axis = 0) 
            #fft_skew1 = skew(np_fft1, axis = 0) #asymmetric
            #fft_kurtosis1= kurtosis(np_fft1, axis = 0) #shape
            #fft_time_skew1 = DSPUtils.stats_describe(skew(np_fft1, axis = 1)) #asymmetric
            #fft_time_kurtosis1= DSPUtils.stats_describe(kurtosis(np_fft1, axis = 1)) #shape
            fft_time_max1 = DSPUtils.stats_describe(np.max(np_fft1, axis = 1))
            fft_time_mean1 = DSPUtils.stats_describe(np.mean(np_fft1, axis = 1))
            fft_time_std1 = DSPUtils.stats_describe(np.std(np_fft1, axis = 1))
            magnitude1 = np.abs(np_fft1)
            frequency_bins = np.fft.fftfreq(len(np_fft1))
            spectral_energy1 = np.sum(magnitude1**2)
            #dominant_frequency1 = np.abs(frequency_bins[np.argmax(magnitude1)])
            #spectral_entropy1 = -np.sum(magnitude1**2 * np.log(magnitude1**2))


            np_fft2 = np.array(fft_windows2)
            fft_len2 = [len(fft_windows2)]
            fft_mean2 = np.mean(np_fft2, axis = 0)
            fft_quantile1st2 = np.quantile(np_fft2, 0.25, axis = 0)
            fft_quantile3th2 = np.quantile(np_fft2, 0.75, axis = 0)
            fft_median2 = np.median(np_fft2, axis = 0)
            fft_max2 = np.amax(np_fft2, axis = 0)
            fft_std2 = np.std(np_fft2, axis = 0) #asymmetric
            fft_hmean2 =hmean(np_fft2, axis = 0) #asymmetric
            fft_moment2 = moment(np_fft2, axis = 0) 
            #fft_skew2 = skew(np_fft2, axis = 0) #asymmetric
            #fft_kurtosis2= kurtosis(np_fft2, axis = 0) #shape
            #fft_time_skew2 = DSPUtils.stats_describe(skew(np_fft2, axis = 1)) #asymmetric
            #fft_time_kurtosis2= DSPUtils.stats_describe(kurtosis(np_fft2, axis = 1)) #shape
            fft_time_max2 = DSPUtils.stats_describe(np.max(np_fft2, axis = 1))
            fft_time_mean2 = DSPUtils.stats_describe(np.mean(np_fft2, axis = 1))
            fft_time_std2 = DSPUtils.stats_describe(np.std(np_fft2, axis = 1))
            magnitude2 = np.abs(np_fft2)
            frequency_bins = np.fft.fftfreq(len(np_fft2))
            spectral_energy2 = np.sum(magnitude2**2)
            #dominant_frequency2 = np.abs(frequency_bins[np.argmax(magnitude2)])
            #spectral_entropy2 = -np.sum(magnitude2**2 * np.log(magnitude2**2))



            np_fft3 = np.array(fft_windows3)
            fft_len3 = [len(fft_windows3)]
            fft_mean3 = np.mean(np_fft3, axis = 0)
            fft_quantile1st3 = np.quantile(np_fft3, 0.25, axis = 0)
            fft_quantile3th3 = np.quantile(np_fft3, 0.75, axis = 0)
            fft_median3 = np.median(np_fft3, axis = 0)
            fft_max3 = np.amax(np_fft3, axis = 0)
            fft_std3 = np.std(np_fft3, axis = 0) #asymmetric
            fft_hmean3 =hmean(np_fft3, axis = 0) #asymmetric
            fft_moment3 = moment(np_fft3, axis = 0) 
            #fft_skew3 = skew(np_fft3, axis = 0) #asymmetric
            #fft_kurtosis3= kurtosis(np_fft3, axis = 0) #shape
            #fft_time_skew3 = DSPUtils.stats_describe(skew(np_fft3, axis = 1)) #asymmetric
            #fft_time_kurtosis3= DSPUtils.stats_describe(kurtosis(np_fft3, axis = 1)) #shape
            fft_time_max3 = DSPUtils.stats_describe(np.max(np_fft3, axis = 1))
            fft_time_mean3 = DSPUtils.stats_describe(np.mean(np_fft3, axis = 1))
            fft_time_std3 = DSPUtils.stats_describe(np.std(np_fft3, axis = 1))
            magnitude3 = np.abs(np_fft3)
            frequency_bins = np.fft.fftfreq(len(np_fft3))
            spectral_energy3 = np.sum(magnitude3**2)
            #dominant_frequency3 = np.abs(frequency_bins[np.argmax(magnitude3)])
            #spectral_entropy3 = -np.sum(magnitude3**2 * np.log(magnitude3**2))



            np_fft4 = np.array(fft_windows4)
            fft_len4 = [len(fft_windows4)]
            fft_mean4 = np.mean(np_fft4, axis = 0)
            fft_quantile1st4 = np.quantile(np_fft4, 0.25, axis = 0)
            fft_quantile3th4 = np.quantile(np_fft4, 0.75, axis = 0)
            fft_median4 = np.median(np_fft4, axis = 0)
            fft_max4 = np.amax(np_fft4, axis = 0)
            fft_std4 = np.std(np_fft4, axis = 0) #asymmetric
            fft_hmean4 =hmean(np_fft4, axis = 0) #asymmetric
            fft_moment4 = moment(np_fft4, axis = 0) 
            #fft_skew4 = skew(np_fft4, axis = 0) #asymmetric
            #fft_kurtosis4= kurtosis(np_fft4, axis = 0) #shape
            #fft_time_skew4 = DSPUtils.stats_describe(skew(np_fft4, axis = 1)) #asymmetric
            #fft_time_kurtosis4= DSPUtils.stats_describe(kurtosis(np_fft4, axis = 1)) #shape
            fft_time_max4 = DSPUtils.stats_describe(np.max(np_fft4, axis = 1))
            fft_time_mean4 = DSPUtils.stats_describe(np.mean(np_fft4, axis = 1))
            fft_time_std4 = DSPUtils.stats_describe(np.std(np_fft4, axis = 1))
            magnitude4 = np.abs(np_fft4)
            frequency_bins = np.fft.fftfreq(len(np_fft4))
            spectral_energy4 = np.sum(magnitude4**2)
            #dominant_frequency4 = np.abs(frequency_bins[np.argmax(magnitude4)])
            #spectral_entropy4 = -np.sum(magnitude4**2 * np.log(magnitude4**2))


            #features=np.concatenate(features,abs(np_fft.flatten()))
            #features=np.concatenate(features,abs(np_fft1.flatten()))
            #features=np.concatenate(features,abs(np_fft2.flatten()))
            #features=np.concatenate(features,abs(np_fft3.flatten()))
            #features=np.concatenate(features,abs(np_fft4.flatten()))


            features = np.concatenate([
                    # fft_overall,
                    #abs(np_fft.flatten()),
                    fft_len,
                    fft_max, 
                    fft_mean,
                    fft_quantile1st, 
                    fft_median, 
                    fft_quantile3th,  
                    fft_hmean,
                    fft_moment,
                    #fft_skew, 
                    #fft_kurtosis, 
                    fft_std, 
                    #fft_time_skew,  
                    #fft_time_kurtosis,  
                    fft_time_max,
                    fft_time_mean,
                    fft_time_std,
                    #spectral_energy,
                    #dominant_frequency,
                    #spectral_entropy,

                    #abs(np_fft1.flatten()),
                    fft_len1,
                    fft_max1, 
                    fft_mean1,
                    fft_quantile1st1, 
                    fft_median1, 
                    fft_quantile3th1,  
                    fft_hmean1,
                    fft_moment1,
                    #fft_skew1, 
                    #fft_kurtosis1, 
                    fft_std1, 
                    #fft_time_skew1,  
                    #fft_time_kurtosis1,  
                    fft_time_max1,
                    fft_time_mean1,
                    fft_time_std1,
                    #spectral_energy1,
                    #dominant_frequency1,
                    #spectral_entropy1,

                    #abs(np_fft2.flatten()),
                    fft_len2,
                    fft_max2, 
                    fft_mean2,
                    fft_quantile1st2, 
                    fft_median2, 
                    fft_quantile3th2,  
                    fft_hmean2,
                    fft_moment2,
                    #fft_skew2, 
                    #fft_kurtosis2, 
                    fft_std2, 
                    #fft_time_skew2,  
                    #fft_time_kurtosis2,  
                    fft_time_max2,
                    fft_time_mean2,
                    fft_time_std2,
                    #spectral_energy2,
                    #dominant_frequency2,
                    #spectral_entropy2,

                    #abs(np_fft3.flatten()),
                    fft_len3,
                    fft_max3, 
                    fft_mean3,
                    fft_quantile1st3, 
                    fft_median3, 
                    fft_quantile3th3,  
                    fft_hmean3,
                    fft_moment3,
                    #fft_skew3, 
                    #fft_kurtosis3, 
                    fft_std3, 
                    #fft_time_skew3,  
                    #fft_time_kurtosis3,  
                    fft_time_max3,
                    fft_time_mean3,
                    fft_time_std3,
                    #spectral_energy3,
                    #dominant_frequency3,
                    #spectral_entropy3,

                    #abs(np_fft4.flatten()),
                    fft_len4,
                    fft_max4, 
                    fft_mean4,
                    fft_quantile1st4, 
                    fft_median4, 
                    fft_quantile3th4,  
                    fft_hmean4,
                    fft_moment4,
                    #fft_skew4, 
                    #fft_kurtosis4, 
                    fft_std4, 
                    #fft_time_skew4,  
                    #fft_time_kurtosis4,  
                    fft_time_max4,
                    fft_time_mean4,
                    fft_time_std4,
                    #spectral_energy4,
                    #dominant_frequency4,
                    #spectral_entropy4

 
                    ])
        
        #else: 
        #   features = [fft_quantile1st]

        return features

    @staticmethod
    def extract_feature_from_raw_signal(sig, background_fft_profile):
        WINDOW_SIZE = 512
        SHIFT_SIZE= 128

        filtered_signal = sig
        filtered_signal = filtered_signal
        windows = DSPUtils.apply_sliding_window(filtered_signal, WINDOW_SIZE, SHIFT_SIZE)
        segmented_signal, fft_windows = DSPUtils.segment_along_windows(windows, background_fft_profile, WINDOW_SIZE, SHIFT_SIZE)

        return DSPUtils.extract_feature(segmented_signal, fft_windows, WINDOW_SIZE)

    @staticmethod
    def covert_single_data_to_ts_format(data):
        tsfresh_data_format = []
        index = 0
        time_count = 0
        for value in data:
            tsfresh_data_format.append([index, time_count, value])
            time_count += 1/DSPUtils.SAMPLE_FREQUENCY
        df = pd.DataFrame(data=tsfresh_data_format, columns=['id', 'time', 'value'])
        return df

    @staticmethod
    def covert_all_data_to_ts_format(all_data):
        tsfresh_data_format = []
        y = []
        index = 0
        for  o in all_data:
            for data in all_data[o]:
                time_count = 0
                for value in data:
                    tsfresh_data_format.append([index, time_count, value])
                    time_count += 1/DSPUtils.SAMPLE_FREQUENCY
                y.append(o)
                index+=1
        df = pd.DataFrame(data=tsfresh_data_format, columns=['id', 'time', 'value'])
        return df, y

    @staticmethod
    def compute_relevant_features(all_data):
        df, y = covert_all_data_to_ts_format(all_data)
        features_filtered_direct = extract_relevant_features(df, y, column_id="id", column_sort="time", column_kind="kind", column_value="value", default_fc_parameters=settings)
                
    @staticmethod
    def generate_sine_wave(freq, sample_rate, duration):
        x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = x * freq
        # 2pi because np.sin takes radians
        y = np.sin((2 * np.pi) * frequencies)
        return x, y


if __name__ == '__main__':
    print(DSPUtils.generate_sine_wave(500, 1000, 1))
