import numpy as np
from scipy.stats import moment, beta

def uniform_random(N):
    """Uniform random distribution
    
    :param N: Number of points.
    :type N: int
    """
    burst = np.random.rand(N) - 0.5
    return burst / np.max(np.abs(burst))

def normal_random(N):
    """Normal random distribution.
    
    :param N: Number of points.
    :type N: int
    """
    burst = np.random.randn(N)
    return burst / np.max(np.abs(burst))

def pseudo_random(N):
    """Pseudorandom distribution.

    Magnitudes are 1, phase is random.
    
    :param N: Number of points.
    :type N: int
    """
    R = np.ones(N//2+1,complex)
    R_prand = R * np.exp(1j*np.random.rand(len(R))*2*np.pi)
    burst = np.fft.irfft(R_prand)
    return burst / np.max(np.abs(burst))


def random_gaussian(N, PSD, fs):
    """Stationary Gaussian realization of random process, characterized by PSD. 
    
    Random process is obtained with IFFT of amplitude spectra with random phase [1]. Area under PSD curve represents variance of random process.
    
    :param N: Number of points.
    :type N: int
    :param PSD: one-sided power spectral density [unit^2].
    :type PSD: array
    :param fs: sampling frequency [Hz].
    :type fs: (int,float)
    :returns: stationary Gaussian realization of random process

    References
    ----------
    [1] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
        2005
    """

    ampl_spectra = np.sqrt(PSD * N * fs / 2)  # amplitude spectra    
    ampl_spectra_random = ampl_spectra * np.exp(1j * np.random.rand(len(PSD)) * 2 * np.pi) #amplitude spectra, random phase
    burst = np.fft.irfft(ampl_spectra_random) # time signal
    return burst


def burst_random(N, A=1., ratio=0.5, distribution='uniform', n_bursts=1, periodic_bursts=True):
    """
    Generate a zero-mean burst random excitation signal time series.
    
    :param N: Number of time points.
    :param A: Amplitude of the random signal. For 'uniform' distribution, this
        is the peak-to-peak amplitude, for 'normal' distribution this is the RMS.
    :param ratio: The ratio of burst legth ot the total legth of the time series.
    :param distribution: 'uniform', 'normal' or 'pseudorandom'. Defaults to 'uniform'.
    :param n_bursts: Number of burst repetition. The output time series will
        have `N*n_bursts` points. Defaults to 1.
    :param periodic_bursts: If True, bursts are periodically repeated `n_bursts` times, 
        otherwise a uniquely random burst is generated for each repetition. 
        Defaults to True.
    :returns: Burst random signal time series.
    """
    
    if not isinstance(n_bursts, int) or n_bursts < 1:
        raise ValueError('`n_bursts` must be a positive integer!')
    
    bursts = []
    
    if not periodic_bursts:
        n = n_bursts
    else:
        n = 1
        
    for _ in range(n):
        if distribution == 'uniform':
            br = uniform_random(N) * A
        elif distribution == 'normal':
            br = normal_random(N) * A
        elif distribution == 'pseudorandom':
            br = pseudo_random(N) * A
        else:
            raise ValueError("Set `ditribution` either to 'normal', 'uniform' or 'periodic'.")

        if ratio != 1.:
            N_zero = int(np.floor(N * (1-ratio)))
            br[-N_zero:] = 0.
        
        bursts.append(br)
    bursts = np.asarray(bursts).flatten()
    
    if periodic_bursts:
        if n_bursts > 1:
            bursts = np.tile(bursts, n_bursts)
        
    return bursts


def sweep(time, phi=0, f_start=1, sweep_rate=None, f_stop=None, mode='linear'):
    """
    TBA
    """
    if sweep_rate is None:
        if not f_stop is None:
            T = time[-1] - time[0]
            sweep_rate = _sweep_rate(T, f_start, f_stop, mode)
        else:
            raise ValueError('`sweep_rate` is not given, please supply `f_stop`.')
    
    if mode == 'linear':
        phase_t = 2*np.pi * (sweep_rate*0.5*time**2 + f_start*time)
    elif mode == 'logarithmic':
        phase_t = 2*np.pi * 60*f_start/(sweep_rate*np.log(2)) * (2**(sweep_rate*time/60) - 1)
    else:
        raise ValueError('Invalid sweep mode `{mode}`.')
    
    s = np.sin(phase_t + phi)
    return s


def _sweep_rate(T, f_start, f_stop, mode='linear'):
    """
    Calculate the sweep rate given the time difference, initial and end
    frequency values and sweep mode. For internal use by `sweep`.
    """
    if mode == 'linear':
        sweep_rate = (f_stop - f_start) / T # Hz/s
    elif mode == 'logarithmic':
        sweep_rate = np.log((f_stop/f_start)**(60/T/np.log(2))) # octaves/min
    else:
        raise ValueError('Invalid sweep mode `{mode}`.')
    return sweep_rate


def stationary_nongaussian_signal(N, PSD, fs, s_k = 0, k_u = 3, mean = 0):
    """
     Stationary non-Gaussian realization of random process. 
    
    Random process is obtained with IFFT of amplitude spectra with random phase [1]. Non-Gaussianity is obtained by Winterstein polynomials [2]. 

    :param N: - number of data points in returned signal
    :type N: int
    :param PSD: - one-sided PSD
    :type PSD:  array
    :param fs: - sampling period
    :type fs: (int, float)
    :param s_k: - skewness of returned signal
    :type s_k: (int, float)
    :param k_u: - kurtossis of returned signal
    :type k_u: (int, float)
    :param mean: - mean value of returned signal
    :type mean: (int, float)
    :returns: stationary non-Gaussian realization of random process.

    References
    ----------
    [1] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
        2005
    [2] Steven R. Winterstein. Nonlinear vibration models for extremes and fatigue. ASCE Journal of Engineering
        Mechanics, 114:1772–1790, 1988.

    """
    x = random_gaussian(N, PSD, fs) #gaussian random process

    h_4 = (np.sqrt(1 + 1.5*(k_u -3)) - 1)/18 #parameter h4 [2]
    h_3 = s_k/(6*(1 + 6*h_4)) ##parameter h3 [2]
    Κ = 1/np.sqrt(1 + 2*h_3**2 + 6*h_4**2) #parameter K [2]
    σ_x = np.std(x) #standard deviation of gaussian process
    nongaussian_signal = mean + Κ*(x/σ_x + h_3*(x/σ_x - 1) + h_4*((x/σ_x)**3 - 3*x/σ_x)) #[2]

    return nongaussian_signal


def _get_nonstationary_signal_psd(N, PSD, PSD_modulating, fs, delta_m = 1, p = 1):
    """
    Non-stationary non-Gaussian realization of random process.

    Non-stationarity random process is obtained by amplitude modulation of Gaussian random process[1]. Carrier signal is Gaussian random process, obtained with IFFT of amplitude spectra with random phase [2]. Modulating signal is generated on PSD basis [3]. For internal use by `nonstationary_signal_psd`.

    :param N: - number of data points in returned signal
    :type N: (int, float)
    :param PSD: - one-sided power spectral density of carrier signal
    :type PSD: array
    :param PSD_modulating: - one-sided power spectral density of modulating signal
    :type PSD_modulating: array
    :param fs: - sampling period
    :type fs: (int, float)
    :param delta_m: - offset 
    :type delta_m: (int, float)
    :param p: - exponent
    :type p: (int, float)
    :returns: non-stationary non-Gaussian realization of random process.

    References
    ----------
    [1] Frederic Kihm, Stephen A. Rizzi, N. S. Ferguson, and Andrew Halfpenny. Understanding how kurtosis is
        transferred from input acceleration to stress response and it’s influence on fatigue life. In Proceedings of the
        XI International Conference on Recent Advances in Structural Dynamics, Pisa, Italy, 07 2013.
    [2] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
        2005
    [3] Arvid Trapp, Mafake James Makua, and PeterWolfsteiner. Fatigue assessment of amplitude-modulated nonstationary
        random vibration loading. Procedia Structural Integrity, 17:379—-386, 2019.

    """
    stationary_signal = random_gaussian(N, PSD, fs) # gaussian random process, carrier
    modulating_signal = random_gaussian(N, PSD_modulating, fs) # gaussian random process,  modulating signal
    
    nonstationary_signal = stationary_signal*(np.abs(modulating_signal)**p + delta_m) # [3]
    nonstationary_signal = nonstationary_signal/np.std(nonstationary_signal) # non-stationary signal 
    
    return nonstationary_signal


def nonstationary_signal_psd(N, PSD, PSD_modulating, fs, delta_m_list, p_list, k_u = 3):
    """
    Non-stationary non-Gaussian realization of random process.
    
    Non-stationarity random process is obtained by amplitude modulation of Gaussian random process[1]. Gaussian random process is obtained with IFFT of amplitude spectra with random phase [2]. Modulating signal is generated on PSD basis [3]. The desired kurtosis k_u is obtained by iteration over lists delta_m_list and p_list.

    :param N: - number of data points in returned signal
    :type N: (int, float)
    :param PSD: - one-sided power spectral density of carrier signal
    :type PSD: array
    :param PSD_modulating: - one-sided power spectral density of modulating signal
    :type PSD_modulating: array
    :param fs: - sampling period
    :type fs: (int, float)
    :param delta_m_list: - offset list
    :type delta_m_list: list of floats
    :param p_list: - exponents list
    :type p_list: list of floats
    :returns: non-stationary non-Gaussian realization of random process.

    References
    ----------
    [1] Frederic Kihm, Stephen A. Rizzi, N. S. Ferguson, and Andrew Halfpenny. Understanding how kurtosis is
        transferred from input acceleration to stress response and it’s influence on fatigue life. In Proceedings of the
        XI International Conference on Recent Advances in Structural Dynamics, Pisa, Italy, 07 2013.
    [2] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
        2005
    [3] Arvid Trapp, Mafake James Makua, and PeterWolfsteiner. Fatigue assessment of amplitude-modulated nonstationary
        random vibration loading. Procedia Structural Integrity, 17:379—-386, 2019.

    """
    nonstationary_signals_tmp = {} # temporary signals dict
    delta_k_u_dict = {} # difference of actual and targeted kurtosis     

    for p in p_list:     
        for delta_m in delta_m_list: 
            sig_tmp = _get_nonstationary_signal_psd(N, PSD, PSD_modulating, fs, delta_m=delta_m, p = p)
            nonstationary_signals_tmp[f'ind_p={p}, ind_m={delta_m}'] = sig_tmp
            k_u_tmp = moment(sig_tmp, 4)/(moment(sig_tmp, 2)**2)
            delta_k_u_dict[f'ind_p={p}, ind_m={delta_m}'] = np.abs(k_u - k_u_tmp)
            
    min_key = min(delta_k_u_dict, key=delta_k_u_dict.get) 
    
    return nonstationary_signals_tmp[min_key]