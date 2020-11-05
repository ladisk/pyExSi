import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import signal_generation as sg


#IMPACT PULSE 
length = 300
N = 3*length
n_start = 90
amplitude = 3
pulse_rectangular = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='boxcar')
pulse_triangular = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='triang')
pulse_blackman = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='blackman')
pulse_hamming = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='hamming')
pulse_hann = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='hann')
pulse_bartlett = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='bartlett')
pulse_flattop = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='flattop')
pulse_parzen = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='parzen')
pulse_bohman = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='bohman')
pulse_blackmanharris = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='blackmanharris')
pulse_nuttall = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='nuttall')
pulse_barthann = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='barthann')
pulse_kaiser = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window=('kaiser',.1))
pulse_exponential = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window=('exponential',None,10))
pulse_sawtooth = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, window='sawtooth')

plt.plot(pulse_rectangular, '-', label='rectangular')
plt.plot(pulse_triangular, '-', label = 'triangular')
plt.plot(pulse_blackman, '-', label = 'blackman')
plt.plot(pulse_hamming, '-', label = 'hamming')
plt.plot(pulse_hann, '-', label = 'hann')
plt.plot(pulse_bartlett, '-', label = 'bartlett')
plt.plot(pulse_flattop, '-', label = 'flattop')
plt.plot(pulse_parzen, '-', label = 'parzen')
plt.plot(pulse_bohman, '-', label = 'bohman')
plt.plot(pulse_blackmanharris, '-', label = 'blackmanharris')
plt.plot(pulse_nuttall, '-', label = 'nuttall')
plt.plot(pulse_barthann, '-', label = 'barthann')
plt.plot(pulse_kaiser, '-', label = 'kaiser')
plt.plot(pulse_exponential, '-', label = 'exponential')
plt.plot(pulse_sawtooth, '-', label = 'sawtooth')
plt.legend(loc='upper right')
plt.show()




#RANDOM SIGNALSS
N = 2**16 # number of data points of time signal
fs = 1024 # sampling frequency [Hz]
t = np.arange(0,N)/fs # time vector

# define frequency vector and one-sided flat-shaped PSD
M = int(N/2 + 1) # number of data points of frequency vector
f = np.arange(0, M, 1) * fs / N # frequency vector
f_min = 50 # PSD upper frequency limit  [Hz]
f_max = 100 # PSD lower frequency limit [Hz]
PSD = sg.get_psd(f, f_min, f_max) # one-sided flat-shaped PSD
plt.plot(f,PSD)
plt.xlim(0,200)
plt.show()

#get gaussian stationary signal
gausian_signal = sg.random_gaussian(N,PSD, fs)
#calculate kurtosis 
k_u_stationary = sg.get_kurtosis(gausian_signal)

#get non-gaussian stationary signal, with kurtosis k_u=10
nongausian_signal = sg.stationary_nongaussian_signal(N, PSD, fs, k_u=10)
#calculate kurtosis
k_u_stationary_nongaussian = sg.get_kurtosis(nongausian_signal)

#get non-gaussian non-stationary signal, with kurtosis k_u=10
#a) amplitude modulation, modulating signal defined by PSD
PSD_modulating = sg.get_psd(f, f_low=1, f_high=10) 
plt.plot(f,PSD)
plt.plot(f,PSD_modulating)
plt.xlim(0,200)
plt.show()
#define array of parameters delta_m and p
delta_m_list = np.arange(.1,2.1,.1) 
p_list = np.arange(.1,2.1,.1)
#get signal 
nongausian_nonsttaionary_signal_psd = sg.nonstationary_signal(N,PSD,fs,k_u=10,modulating_signal=('PSD',PSD_modulating),
                                                        param1_list=delta_m_list,param2_list=p_list)
#calculate kurtosis 
k_u_nonstationary_nongaussian_psd = sg.get_kurtosis(nongausian_nonsttaionary_signal_psd)

#b) amplitude modulation, modulating signal defined by cubis spline intepolation. Points are based on beta distribution
#Points are separated by delta_n = 2**8 samples (at fs=2**10)
delta_n = 2**8
#define array of parameters alpha and beta
alpha_list = np.arange(1,7,1)
beta_list = np.arange(1,7,1)
#get signal 
nongausian_nonsttaionary_signal_beta = sg.nonstationary_signal(N,PSD,fs,k_u=10,modulating_signal=('CSI',delta_n),
                                                        param1_list=alpha_list,param2_list=beta_list)
#calculate kurtosis 
k_u_nonstationary_nongaussian_beta = sg.get_kurtosis(nongausian_nonsttaionary_signal_beta)

#Plot
t_indx = np.logical_and(t>0, t<1)
plt.plot(t[t_indx],gausian_signal[t_indx], label = 'gaussian')
plt.plot(t[t_indx],nongausian_signal[t_indx], label = 'non-gaussian stationary')
plt.plot(t[t_indx],nongausian_nonsttaionary_signal_psd[t_indx], label = 'non-gaussian non-stationary (psd)')
plt.plot(t[t_indx],nongausian_nonsttaionary_signal_beta[t_indx], label = 'non-gaussian non-stationary (beta)')
plt.legend()
plt.show()

print(f'kurtosis of stationary Gaussian signal:{k_u_stationary:.3f}')
print(f'kurtosis of stationary non-Gaussian signal:{k_u_stationary_nongaussian:.3f}')
print(f'kurtosis of non-stationary non-Gaussian signal (psd):{k_u_nonstationary_nongaussian_psd:.3f}')
print(f'kurtosis of non-stationary non-Gaussian signal (beta):{k_u_nonstationary_nongaussian_beta:.3f}')