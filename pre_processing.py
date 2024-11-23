from scipy.signal import welch, butter, filtfilt, periodogram, spectrogram, freqz
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.io import loadmat

from biosppy.signals.ecg import ecg

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import nolds
import glob
import os

def calculate_nni_ms(signal, sampling_rate=100):
    rpeaks_index = ecg(signal=signal, sampling_rate=sampling_rate, show=False)[2]
    rpeaks_ms = rpeaks_index/sampling_rate * 1000.0
    rri_ms = np.diff(rpeaks_ms)
    nni_ms = rri_ms
    return nni_ms


def get_time_domain_features(nni_ms):
    diff_nni_ms = np.diff(nni_ms)
    hr = 60000.0/nni_ms
    
    time_domain_features = {
        'Mean_NN': np.mean(nni_ms),
        'Mean_HR': np.mean(hr),
        'STD_HR': np.std(hr, ddof=1),
        'SDSD': np.std(diff_nni_ms, ddof=1),
        'SDNN': np.std(nni_ms, ddof=1),
        'RMSSD': np.sqrt(np.mean(np.square(diff_nni_ms))),
        'NN50': np.sum(np.abs(diff_nni_ms) > 50),
        'pNN50': 100 * np.sum(np.abs(diff_nni_ms) > 50) / (len(nni_ms)-1)
    }
    return time_domain_features


def get_frequency_domain_features(nni_ms, fs=4, welch_method=False):
    t = np.cumsum(nni_ms)
    t -= t[0]
    f_interp1d = interp1d(t, nni_ms, kind='cubic')
    t_interpolated = np.arange(0, t[-1], 1000.0/fs)
    nni_interpolated = f_interp1d(t_interpolated)

    if welch_method==True:
        f, pxx = welch(x=nni_interpolated, fs=fs, window='bartlett', nfft=2**12) #, nperseg=len(nni_interpolated), noverlap=0 (similar to periodogram)
    else:
        f, pxx = periodogram(x=nni_interpolated, fs=fs, window='bartlett', nfft=2**12)
    
    condition_vlf = (f >= 0.003) & (f < 0.04)
    condition_lf =  (f >= 0.04) & (f < 0.15)
    condition_hf =  (f >= 0.15) & (f < 0.4)
    
    power_vlf = trapz(pxx[condition_vlf], f[condition_vlf])
    power_lf =  trapz(pxx[condition_lf], f[condition_lf])
    power_hf =  trapz(pxx[condition_hf], f[condition_hf])
    
    frequency_domain_features = {
        'Power_VLF': power_vlf,
        'Power_LF': power_lf,
        'Power_HF': power_hf,
        'Power_Total': power_vlf + power_lf + power_hf,
        'LF_HF_Ratio': (power_lf/power_hf), 
        'Peak_VLF': f[condition_vlf][np.argmax(pxx[condition_vlf])],
        'Peak_LF': f[condition_lf][np.argmax(pxx[condition_lf])],
        'Peak_HF': f[condition_hf][np.argmax(pxx[condition_hf])],
        'Fraction_LF': 100 * power_lf / (power_hf + power_lf),
        'Fraction_HF': 100 * power_hf / (power_hf + power_lf),
    }
    return frequency_domain_features


def get_non_linear_features(nni_ms):
    # # From SDNN and SDSD
    # diff_nni_ms = np.diff(nni_ms)
    # SDNN = np.std(nni_ms, ddof=1)
    # SDSD = np.std(diff_nni_ms, ddof=1)
    # sd1 = np.sqrt(0.5 * np.power(SDSD, 2))
    # sd2 = np.sqrt(2 * np.power(SDNN, 2) - 0.5 * np.power(SDSD, 2))

    # From point-care plot
    x1 = nni_ms[:-1]
    x2 = nni_ms[1:]
    sd1 = np.sqrt(0.5) * np.std((x2 - x1), ddof=1)
    sd2 = np.sqrt(0.5) * np.std((x2 + x1), ddof=1)

    non_linear_features = {
        'SD1': sd1,
        'SD2': sd2,
        'SD1_SD2_Ratio': sd1/sd2,
        'S': np.pi * sd1 * sd2,
        'Alpha1': nolds.dfa(nni_ms, range(4, 17), debug_data=True, overlap=False)[0],
        'Alpha2': nolds.dfa(nni_ms, range(17, 65), debug_data=True, overlap=False)[0],
    }
    return non_linear_features


def get_down_sampled_signal(signal, down_sampling_rate, sampling_rate=100, order=5, show=False):
    b, a = butter(order, (sampling_rate/down_sampling_rate)/2, btype='lowpass', analog=False, fs=sampling_rate)
    filtered_signal = filtfilt(b, a, signal)
    down_sampled_signal = filtered_signal[::down_sampling_rate]

    if show:
        w, h = freqz(b, a, worN=2000)
        plt.plot((sampling_rate * 0.5 / np.pi) * w, abs(h), color="#000000")
        plt.show()

    return down_sampled_signal


def segmentation_2min(trials_addresses, desired_address, range_sensors=list(range(48))+list(range(54, 60)), down_sampling_rate=5, nperseg=80, noverlap=40):
    
    if len(glob.glob(desired_address[:-30]))==0:
        os.mkdir(desired_address[:-30])

    if len(glob.glob(desired_address[:-10]))==0:
        os.mkdir(desired_address[:-10])
    
    if len(glob.glob(desired_address))==0:
        os.mkdir(desired_address)
    else:
        for files in os.listdir(desired_address):
            path = os.path.join(desired_address, files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
    
    hrv = pd.DataFrame()
    
    for trial_address in trials_addresses:
        suffix = trial_address.split('/')[-1].split('_')[0] + '_' + trial_address.split('/')[-1].split('_')[-1].split('.')[0]
        
        ecg = loadmat(trial_address)['mp36_s'][2]
        vicons_signals = loadmat(trial_address)['vicon_s']
        range_without_nan = ecg.shape[0]
        
        for r in range_sensors:
            if (np.count_nonzero(np.isnan(vicons_signals[r]) != 0)):
                first_nan_index = np.where(np.isnan(vicons_signals[r]))[0][0]
                if(first_nan_index < range_without_nan):
                    range_without_nan = first_nan_index
        
        for j in range(range_without_nan//12000):
            motions_spectograms = list()
            range_segmentation = (j*12000, (j+1)*12000)
            signal_ecg = ecg[range_segmentation[0] : range_segmentation[1]]
            nni_ms = calculate_nni_ms(signal_ecg)
            resualt = get_time_domain_features(nni_ms)
            resualt.update(get_frequency_domain_features(nni_ms))
            resualt.update(get_non_linear_features(nni_ms))
            hrv[suffix + '_P' + str(j+1)] = pd.DataFrame.from_dict(resualt, orient='index')
            
            for k in range_sensors:
                signal = vicons_signals[k][range_segmentation[0] : range_segmentation[1]]
                down_sampled_signal = get_down_sampled_signal(signal, down_sampling_rate)
                fxx, txx, sxx = spectrogram(x=down_sampled_signal, fs=20, window='bartlett', nperseg=nperseg, noverlap=noverlap)
                motions_spectograms.append(sxx)
                
            motions_spectograms = np.stack(motions_spectograms, axis=0)
            np.save(desired_address + '/' + suffix + '_P' + str(j+1) + '.npy', motions_spectograms.transpose(1,2,0))

    hrv = hrv.transpose()
    hrv.to_csv(desired_address + '/HRV.csv', index=True, index_label='Trial Name')
    np.save(desired_address + '/fxx.npy', fxx)
    np.save(desired_address + '/txx.npy', txx)


def augmentation_2min_shift_XSec(trials_addresses, desired_address, range_sensors=list(range(48))+list(range(54, 60)), x=2, down_sampling_rate=5, nperseg=80, noverlap=40):
    
    if len(glob.glob(desired_address[:-30]))==0:
        os.mkdir(desired_address[:-30])

    if len(glob.glob(desired_address[:-10]))==0:
        os.mkdir(desired_address[:-10])

    if len(glob.glob(desired_address))==0:
        os.mkdir(desired_address)
    else:
        for files in os.listdir(desired_address):
            path = os.path.join(desired_address, files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
    
    xsec = x * 100
    hrv = pd.DataFrame()
    
    for trial_address in trials_addresses:
        suffix = trial_address.split('/')[-1].split('_')[0] + '_' + trial_address.split('/')[-1].split('_')[-1].split('.')[0]
        
        ecg = loadmat(trial_address)['mp36_s'][2]
        vicons_signals = loadmat(trial_address)['vicon_s']
        range_without_nan = ecg.shape[0]
        
        for r in range_sensors:
            if (np.count_nonzero(np.isnan(vicons_signals[r]) != 0)):
                first_nan_index = np.where(np.isnan(vicons_signals[r]))[0][0]
                if(first_nan_index < range_without_nan):
                    range_without_nan = first_nan_index
                                
        for j in range(int(range_without_nan>12000) + ((range_without_nan-12000)//xsec)):
            motions_spectograms = list()
            range_segmentation = (j*xsec, j*xsec+12000)
            signal_ecg = ecg[range_segmentation[0] : range_segmentation[1]]
            nni_ms = calculate_nni_ms(signal_ecg)
            resualt = get_time_domain_features(nni_ms)
            resualt.update(get_frequency_domain_features(nni_ms))
            resualt.update(get_non_linear_features(nni_ms))
            hrv[suffix + '_P' + str(j+1)] = pd.DataFrame.from_dict(resualt, orient='index')
            
            for k in range_sensors:
                signal = vicons_signals[k][range_segmentation[0] : range_segmentation[1]]
                down_sampled_signal = get_down_sampled_signal(signal, down_sampling_rate)
                fxx, txx, sxx = spectrogram(x=down_sampled_signal, fs=20, window='bartlett', nperseg=nperseg, noverlap=noverlap)
                motions_spectograms.append(sxx)
                
            motions_spectograms = np.stack(motions_spectograms, axis=0)
            np.save(desired_address + '/' + suffix + '_P' + str(j+1) + '.npy', motions_spectograms.transpose(1,2,0))
            
    hrv = hrv.transpose()
    hrv.to_csv(desired_address + '/HRV.csv', index=True, index_label='Trial Name')
    np.save(desired_address + '/fxx.npy', fxx)
    np.save(desired_address + '/txx.npy', txx)


if __name__ =='__main__':
    BASE_PATH = './'
    
    all_trials_address = [BASE_PATH + 'Data/Article/Free Breathing/Free_T{}.mat'.format(i) for i in range(1, 52)]
    assert len(all_trials_address)==51

    segmentation_2min(all_trials_address, BASE_PATH + 'Data/DownSampled_and_HRV/Segmented', nperseg=80, noverlap=40)
    # augmentation_2min_shift_XSec(all_trials_address, BASE_PATH + 'Data/DownSampled_and_HRV/Augmented', x=20 ,nperseg=80, noverlap=40)