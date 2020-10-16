import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import signal_generation as sg


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


#get gaussian stationary signal
gausian_signal = sg.get_gaussian_signal(PSD, fs, N)
#calculate kurtosis 
μ_0 = np.mean(gausian_signal)
μ_2 = stats.moment(gausian_signal, 2)
μ_3 = stats.moment(gausian_signal, 3)
μ_4 = stats.moment(gausian_signal, 4)
k_u_stationary = μ_4/μ_2**2


#get non-gaussian stationary signal, with kurtosis k_u=10
nongausian_signal = sg.get_stationary_nongaussian_signal(PSD, fs, N, k_u=10)
#calculate kurtosis
μ_0 = np.mean(nongausian_signal)
μ_2 = stats.moment(nongausian_signal, 2)
μ_3 = stats.moment(nongausian_signal, 3)
μ_4 = stats.moment(nongausian_signal, 4)
k_u_stationary_nongaussian = μ_4/μ_2**2


#get non-gaussian non-stationary signal, with kurtosis k_u=10
#a) amplitude modulation, modulating signal defined by PSD
PSD_modulating = sg.get_psd(f, f_low=1, f_high=10) 
plt.plot(f,PSD)
plt.plot(f,PSD_modulating)
plt.xlim(0,200)
#define array of parameters delta_m and p
delta_m_list = np.arange(.1,2.1,.5) 
p_list = np.arange(.1,2.1,.5)
#get signal 
nongausian_nonsttaionary_signal_psd = sg.get_nonstationary_signal_psd(PSD,PSD_modulating,fs,N,delta_m_list,p_list, k_u=10, variance=10)
#calculate kurtosis 
μ_0 = np.mean(nongausian_nonsttaionary_signal_psd)
μ_2 = stats.moment(nongausian_nonsttaionary_signal_psd, 2)
μ_3 = stats.moment(nongausian_nonsttaionary_signal_psd, 3)
μ_4 = stats.moment(nongausian_nonsttaionary_signal_psd, 4)
k_u_nonstationary_nongaussian_psd = μ_4/μ_2**2


#b) amplitude modulation, modulating signal defined by cubis spline intepolation. Points are based on beta distribution
#Points are separated by delta_n = 2**8 samples (at fs=2**10)
delta_n = 2**8
#define array of parameters alpha and beta
alpha_list = np.arange(1,7,1)
beta_list = np.arange(1,7,1)
#get signal 
nongausian_nonsttaionary_signal_beta = sg.get_nonstationary_signal_beta(PSD, fs, N, delta_n, alpha_list,beta_list, k_u=5, variance=10)
#calculate kurtosis 
μ_0 = np.mean(nongausian_nonsttaionary_signal_beta)
μ_2 = stats.moment(nongausian_nonsttaionary_signal_beta, 2)
μ_3 = stats.moment(nongausian_nonsttaionary_signal_beta, 3)
μ_4 = stats.moment(nongausian_nonsttaionary_signal_beta, 4)
k_u_nonstationary_nongaussian_beta = μ_4/μ_2**2

#Plot
t_indx = np.logical_and(t>0, t<1)
plt.plot(t[t_indx],gausian_signal[t_indx], label = 'gaussian')
plt.plot(t[t_indx],nongausian_signal[t_indx], label = 'non-gaussian stationary')
plt.plot(t[t_indx],nongausian_nonsttaionary_signal_psd[t_indx], label = 'non-gaussian non-stationary (psd)')
plt.plot(t[t_indx],nongausian_nonsttaionary_signal_beta[t_indx], label = 'non-gaussian non-stationary (beta)')
plt.legend()

print(f'kurtosis of stationary Gaussian signal:{k_u_stationary:.3f}')
print(f'kurtosis of stationary non-Gaussian signal:{k_u_stationary_nongaussian:.3f}')
print(f'kurtosis of non-stationary non-Gaussian signal (psd):{k_u_nonstationary_nongaussian_psd:.3f}')
print(f'kurtosis of non-stationary non-Gaussian signal (beta):{k_u_nonstationary_nongaussian_beta:.3f}')