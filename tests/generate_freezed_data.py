import sys, os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import numpy as np
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import pyExSi as es


def get_nonstationarity_data(show=False):
    test_data = {}

    N = 2 ** 16  # number of data points of time signal
    fs = 1024  # sampling frequency [Hz]
    t = np.arange(0, N) / fs  # time vector

    # define frequency vector and one-sided flat-shaped PSD
    M = N // 2 + 1  # number of data points of frequency vector
    f = np.arange(0, M, 1) * fs / N  # frequency vector
    f_min = 50  # PSD upper frequency limit  [Hz]
    f_max = 100  # PSD lower frequency limit [Hz]
    PSD = es.get_psd(f, f_min, f_max)  # one-sided flat-shaped PSD
    if show:
        plt.plot(f, PSD)
        plt.xlim(0, 200)
        plt.show()

    test_data['N'] = N
    test_data['fs'] = fs
    test_data['freq'] = f
    test_data['PSD'] = PSD
    test_data['f_min'] = f_min
    test_data['f_max'] = f_max

    # Random Generator seed
    seed = 1234
    BitGenerator = np.random.PCG64(seed)
    rg = np.random.default_rng(BitGenerator)

    # get gaussian stationary signal
    gausian_signal = es.random_gaussian(N, PSD, fs, rg=rg)
    # calculate kurtosis
    k_u_stationary = es.get_kurtosis(gausian_signal)

    # get non-gaussian stationary signal, with kurtosis k_u=10
    k_u_target = 10
    rng = np.random.default_rng(seed)
    nongausian_signal = es.stationary_nongaussian_signal(
        N, PSD, fs, k_u=k_u_target, rg=rg
    )
    # calculate kurtosis
    k_u_stationary_nongaussian = es.get_kurtosis(nongausian_signal)

    test_data['seed'] = seed
    test_data['stationary Gaussian'] = gausian_signal
    test_data['stationary nonGaussian'] = nongausian_signal

    # get non-gaussian non-stationary signal, with kurtosis k_u=10
    # a) amplitude modulation, modulating signal defined by PSD
    rng = np.random.default_rng(seed)
    PSD_modulating = es.get_psd(f, f_low=1, f_high=10)
    if show:
        plt.plot(f, PSD)
        plt.plot(f, PSD_modulating)
        plt.xlim(0, 200)
        plt.show()

    test_data['PSD modulating'] = PSD_modulating

    # define array of parameters delta_m and p
    delta_m_list = np.arange(0.1, 2.1, 0.25)
    p_list = np.arange(0.1, 2.1, 0.25)
    # get signal
    nongausian_nonstationary_signal_psd = es.nonstationary_signal(
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
        nongausian_nonstationary_signal_psd
    )

    test_data['delta m list'] = delta_m_list
    test_data['p list'] = p_list
    test_data['nonstationary nonGaussian_PSD'] = nongausian_nonstationary_signal_psd

    # b) amplitude modulation, modulating signal defined by cubis spline intepolation. Points are based on beta distribution
    # Points are separated by delta_n = 2**8 samples (at fs=2**10)
    delta_n = 2 ** 8
    # define array of parameters alpha and beta
    alpha_list = np.arange(1, 10, 1)
    beta_list = np.arange(1, 10, 1)
    # get signal
    nongausian_nonsttaionary_signal_beta = es.nonstationary_signal(
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
        nongausian_nonsttaionary_signal_beta
    )

    test_data['delta_n'] = delta_n
    test_data['alpha list'] = alpha_list
    test_data['beta list'] = beta_list
    test_data['nonstationary nonGaussian CSI'] = nongausian_nonsttaionary_signal_beta

    if show:
        t_indx = np.logical_and(t >= 0, t < 10)
        plt.figure(figsize=(15, 5))
        plt.plot(t[t_indx], gausian_signal[t_indx], label='gaussian')
        plt.plot(t[t_indx], nongausian_signal[t_indx], label='non-gaussian stationary')
        plt.plot(
            t[t_indx],
            nongausian_nonstationary_signal_psd[t_indx],
            label='non-gaussian non-stationary (psd)',
        )
        plt.plot(
            t[t_indx],
            nongausian_nonsttaionary_signal_beta[t_indx],
            label='non-gaussian non-stationary (beta)',
        )
        plt.legend()
        plt.show()

    return test_data


def get_signals_data():
    test_data = {}

    seed = 1234
    BitGenerator = np.random.PCG64(seed)
    rg = np.random.default_rng(BitGenerator)

    N = 2 ** 16

    # uniform random, normal random and pseudo random
    uniform_random = es.uniform_random(N=N, rg=rg)
    normal_random = es.normal_random(N=N, rg=rg)
    pseudo_random = es.pseudo_random(N=N, rg=rg)

    test_data['seed'] = seed
    test_data['N'] = N
    test_data['uniform_random'] = uniform_random
    test_data['normal_random'] = normal_random
    test_data['pseudo_random'] = pseudo_random

    # burst random
    amplitude = 5
    burst_random = es.burst_random(
        N=N, A=amplitude, ratio=0.1, distribution='normal', n_bursts=3, rg=rg
    )
    test_data['burst_random amplitude'] = amplitude
    test_data['burst_random ratio'] = 0.1
    test_data['burst_random distribution'] = 'normal'
    test_data['burst_random n_bursts'] = 3
    test_data['burst_random'] = burst_random

    # sine sweep
    t = np.linspace(0, 10, 1000)
    sweep = es.sine_sweep(time=t, f_start=0, f_stop=5)
    test_data['sweep t'] = t
    test_data['sweep f_start'] = 0
    test_data['sweep f_stop'] = 5
    test_data['sweep'] = sweep

    # impact pulse
    width = 300
    N = 2 * width
    n_start = 90
    amplitude = 3
    pulse_sine = es.impulse(
        N=N, n_start=n_start, width=width, amplitude=amplitude, window='sine'
    )
    pulse_rectangular = es.impulse(
        N=N, n_start=n_start, width=width, amplitude=amplitude, window='boxcar'
    )
    pulse_triangular = es.impulse(
        N=N, n_start=n_start, width=width, amplitude=amplitude, window='triang'
    )
    pulse_exponential = es.impulse(
        N=N,
        n_start=n_start,
        width=width,
        amplitude=amplitude,
        window=('exponential', 0, 10),
    )
    pulse_sawtooth = es.impulse(
        N=N, n_start=n_start, width=width, amplitude=amplitude, window='sawtooth'
    )

    test_data['impulse width'] = width
    test_data['impulse N'] = N
    test_data['impulse n_start'] = n_start
    test_data['impulse amplitude'] = amplitude
    test_data['impulse sine'] = pulse_sine
    test_data['impulse rectangular'] = pulse_rectangular
    test_data['impulse triangular'] = pulse_triangular
    test_data['impulse exponential'] = pulse_exponential
    test_data['impulse sawtooth'] = pulse_sawtooth

    return test_data


if __name__ == "__main__":
    test_data = get_nonstationarity_data(show=False)
    with open(
        'tests/test_data_nonstationarity_human_should_remove_this_if_you_know_what_are_you_doing.pkl',
        'wb',
    ) as the_file:
        pickle.dump(test_data, the_file, protocol=4)

    test_data = get_signals_data()
    with open(
        'tests/test_data_signals_human_should_remove_this_if_you_know_what_are_you_doing.pkl',
        'wb',
    ) as the_file:
        pickle.dump(test_data, the_file, protocol=4)
