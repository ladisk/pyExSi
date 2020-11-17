import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import numpy as np
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import signal_generation as sg


N = 2**16 # number of data points of time signal
fs = 1024 # sampling frequency [Hz]
t = np.arange(0,N)/fs # time vector

# define frequency vector and one-sided flat-shaped PSD
M = N // 2 + 1 # number of data points of frequency vector
f = np.arange(0, M, 1) * fs / N # frequency vector
f_min = 50 # PSD upper frequency limit  [Hz]
f_max = 100 # PSD lower frequency limit [Hz]
PSD = sg.get_psd(f, f_min, f_max) # one-sided flat-shaped PSD
plt.plot(f,PSD)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [Unit**2/Hz]')
plt.xlim(0,200)
plt.show()


#Random Generator seed
seed = 0

#get gaussian stationary signal
rng = np.random.default_rng(seed)
gausian_signal = sg.random_gaussian(N, PSD, fs, rng=rng)
#calculate kurtosis 
k_u_stationary = sg.get_kurtosis(gausian_signal)

#get non-gaussian stationary signal, with kurtosis k_u=10
k_u_target = 10
rng = np.random.default_rng(seed)
nongausian_signal = sg.stationary_nongaussian_signal(N, PSD, fs, k_u=k_u_target, rng=rng)
#calculate kurtosis
k_u_stationary_nongaussian = sg.get_kurtosis(nongausian_signal)

#get non-gaussian non-stationary signal, with kurtosis k_u=10
#a) amplitude modulation, modulating signal defined by PSD
rng = np.random.default_rng(seed)
PSD_modulating = sg.get_psd(f, f_low=1, f_high=k_u_target) 
plt.plot(f,PSD)
plt.plot(f,PSD_modulating)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [Unit**2/Hz]')
plt.xlim(0,200)
plt.show()
#define array of parameters delta_m and p
delta_m_list = np.arange(.1,2.1,.1) 
p_list = np.arange(.1,2.1,.1)
#get signal 
nongausian_nonsttaionary_signal_psd = sg.nonstationary_signal(N,PSD,fs,k_u=k_u_target,modulating_signal=('PSD',PSD_modulating),
                                                        param1_list=delta_m_list,param2_list=p_list,seed=seed)
#calculate kurtosis 
k_u_nonstationary_nongaussian_psd = sg.get_kurtosis(nongausian_nonsttaionary_signal_psd)

#b) amplitude modulation, modulating signal defined by cubis spline intepolation. Points are based on beta distribution
#Points are separated by delta_n = 2**8 samples (at fs=2**10)
delta_n = 2**8
#define array of parameters alpha and beta
alpha_list = np.arange(1,10,1)
beta_list = np.arange(1,10,1)
#get signal 
nongausian_nonsttaionary_signal_beta = sg.nonstationary_signal(N,PSD,fs,k_u=k_u_target,modulating_signal=('CSI',delta_n),
                                                        param1_list=alpha_list,param2_list=beta_list,seed=seed)
#calculate kurtosis 
k_u_nonstationary_nongaussian_beta = sg.get_kurtosis(nongausian_nonsttaionary_signal_beta)

#Plot
plt.plot(gausian_signal[:200], label = 'gaussian')
plt.plot(nongausian_signal[:200], label = 'non-gaussian stationary')
plt.plot(nongausian_nonsttaionary_signal_psd[:200], label = 'non-gaussian non-stationary (psd)')
plt.plot(nongausian_nonsttaionary_signal_beta[:200], label = 'non-gaussian non-stationary (beta)')
plt.xlabel('Sample [n]')
plt.ylabel('Signal [Unit]')
plt.legend()
plt.show()

print(f'kurtosis of stationary Gaussian signal:{k_u_stationary:.3f}')
print(f'Desired kurtosis of non-Gaussian signals:{k_u_target:.3f}')
print(f'kurtosis of stationary non-Gaussian signal:{k_u_stationary_nongaussian:.3f}')
print(f'kurtosis of non-stationary non-Gaussian signal (psd):{k_u_nonstationary_nongaussian_psd:.3f}')
print(f'kurtosis of non-stationary non-Gaussian signal (beta):{k_u_nonstationary_nongaussian_beta:.3f}')