import sys, os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import numpy as np
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import pyExSi as es


N = 2 ** 16  # number of data points of time signal
fs = 1024  # sampling frequency [Hz]
t = np.arange(0, N) / fs  # time vector

# define frequency vector and one-sided flat-shaped PSD
M = N // 2 + 1  # number of data points of frequency vector
freq = np.arange(0, M, 1) * fs / N  # frequency vector
freq_lower = 50  # PSD lower frequency limit  [Hz]
freq_upper = 100  # PSD upper frequency limit [Hz]
PSD = es.get_psd(freq, freq_lower, freq_upper)  # one-sided flat-shaped PSD
plt.plot(freq, PSD)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [Unit**2/Hz]')
plt.xlim(0, 200)
plt.show()


# Random Generator seed
seed = 1234
rg = np.random.default_rng(seed)

# get gaussian stationary signal
gaussian_signal = es.random_gaussian(N, PSD, fs, rg=rg)
# calculate kurtosis
k_u_stationary = es.get_kurtosis(gaussian_signal)

# get non-gaussian stationary signal, with kurtosis k_u=10
k_u_target = 10
rg = np.random.default_rng(seed)
nongaussian_signal = es.stationary_nongaussian_signal(N, PSD, fs, k_u=k_u_target, rg=rg)
# calculate kurtosis
k_u_stationary_nongaussian = es.get_kurtosis(nongaussian_signal)

# get non-gaussian non-stationary signal, with kurtosis k_u=10
# a) amplitude modulation, modulating signal defined by PSD
PSD_modulating = es.get_psd(freq, freq_lower=1, freq_upper=10)
plt.plot(freq, PSD, label='PSD, carrier signal')
plt.plot(freq, PSD_modulating, label='PSD, modulating signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [Unit**2/Hz]')
plt.xlim(0, 200)
plt.legend()
plt.show()
# define array of parameters delta_m and p
delta_m_list = np.arange(0.1, 2.1, 0.5)
p_list = np.arange(0.1, 2.1, 0.5)
# get signal
nongaussian_nonstationary_signal_psd = es.nonstationary_signal(
    N,
    PSD,
    fs,
    k_u=k_u_target,
    modulating_signal=('PSD', PSD_modulating),
    param1_list=delta_m_list,
    param2_list=p_list,
    seed=seed,
)
# calculate kurtosis
k_u_nonstationary_nongaussian_psd = es.get_kurtosis(
    nongaussian_nonstationary_signal_psd
)

# b) amplitude modulation, modulating signal defined by cubis spline intepolation. Points are based on beta distribution
# Points are separated by delta_n = 2**8 samples (at fs=2**10)
delta_n = 2 ** 10
# define array of parameters alpha and beta
alpha_list = np.arange(1, 10, 1)
beta_list = np.arange(1, 10, 1)
# get signal
nongaussian_nonstationary_signal_beta = es.nonstationary_signal(
    N,
    PSD,
    fs,
    k_u=k_u_target,
    modulating_signal=('CSI', delta_n),
    param1_list=alpha_list,
    param2_list=beta_list,
    seed=seed,
)
# calculate kurtosis
k_u_nonstationary_nongaussian_beta = es.get_kurtosis(
    nongaussian_nonstationary_signal_beta
)

# Plot
plt.plot(gaussian_signal[:200], label='Gaussian')
plt.plot(nongaussian_signal[:200], label='non-Gaussian stationary')
plt.plot(
    nongaussian_nonstationary_signal_psd[:200],
    label='non-Gaussian non-stationary (PSD)',
)
plt.plot(
    nongaussian_nonstationary_signal_beta[:200],
    label='non-Gaussian non-stationary (CSI)',
)
plt.xlabel('Sample [n]')
plt.ylabel('Signal [Unit]')
plt.legend()
plt.show()

print(f'kurtosis of stationary Gaussian signal:{k_u_stationary:.3f}')
print(f'Desired kurtosis of non-Gaussian signals:{k_u_target:.3f}')
print(f'kurtosis of stationary non-Gaussian signal:{k_u_stationary_nongaussian:.3f}')
print(
    f'kurtosis of non-stationary non-Gaussian signal (psd):{k_u_nonstationary_nongaussian_psd:.3f}'
)
print(
    f'kurtosis of non-stationary non-Gaussian signal (beta):{k_u_nonstationary_nongaussian_beta:.3f}'
)