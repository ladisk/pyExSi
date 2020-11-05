import numpy as np
from scipy.stats import moment, beta 
from scipy.interpolate import CubicSpline
from scipy import signal

def uniform_random(N):
    """
    Uniform random distribution
    
    :param N: Number of points.
    :type N: int
    """
    burst = np.random.rand(N) - 0.5
    return burst / np.max(np.abs(burst))


def normal_random(N):
    """
    Normal random distribution.
    
    :param N: Number of points.
    :type N: int
    """
    burst = np.random.randn(N)
    return burst / np.max(np.abs(burst))


def pseudo_random(N):
    """
    Pseudorandom distribution.

    Magnitudes are 1, phase is random.
    
    :param N: Number of points.
    :type N: int
    """
    R = np.ones(N//2+1,complex)
    R_prand = R * np.exp(1j*np.random.rand(len(R))*2*np.pi)
    burst = np.fft.irfft(R_prand)
    return burst / np.max(np.abs(burst))


def random_gaussian(N, PSD, fs):
    """
    Stationary Gaussian realization of random process, characterized by PSD. 
    
    Random process is obtained with IFFT of amplitude spectra with random phase [1]. Area under PSD curve represents variance of random process.
    
    :param N: Number of points.
    :type N: int
    :param PSD: one-sided power spectral density [unit^2].
    :type PSD: array
    :param fs: sampling frequency [Hz].
    :type fs: int,float
    :returns: stationary Gaussian realization of random process

    References
    ----------
    [1] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
        2005

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import signal_generation as sg

    >>> N = 1000 # number of data points of time signal
    >>> fs = 100 # sampling frequency [Hz]
    >>> t = np.arange(0,N)/fs # time vector
    >>> M = N // 2 + 1 # number of data points of frequency vector
    >>> f = np.arange(0, M, 1) * fs / N # frequency vector
    >>> f_min = 10 # PSD upper frequency limit  [Hz]
    >>> f_max = 20 # PSD lower frequency limit [Hz]

    >>> PSD = sg.get_psd(f, f_min, f_max) # one-sided flat-shaped PSD
    >>> x = sg.random_gaussian(N, PSD, fs) 
    >>> plt.plot(t,x)
    >>> plt.xlabel(t [s])
    >>> plt.ylabel(x [unit])
    >>> plt.show()
    """
    ampl_spectra = np.sqrt(PSD * N * fs / 2)  # amplitude spectra    
    ampl_spectra_random = ampl_spectra * np.exp(1j * np.random.rand(len(PSD)) * 2 * np.pi) #amplitude spectra, random phase
    burst = np.fft.irfft(ampl_spectra_random) # time signal
    return burst

def impact_pulse(N, n_start, length, amplitude = 1., window = 'half-sine'):
    """
    Impact pulse.

    :param N: number of points in time signal.
    :type N: int
    :param length: length of pulse.
    :type length: int
    :param amplitude: amplitude of pulse.
    :type amplitude: float
    :param window:  The type of window to create. See scipy.signal.windows for more details.
    :type window: string, float, or tuple
    :returns: impact pulse.

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import signal_generation as sg

    >>> N = 1000
    >>> n_start = 90
    >>> amplitude = 3
    >>> x_1 = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude)
    >>> x_2 = sg.impact_pulse(N=N, n_start=n_start, length=length, amplitude=amplitude, shape='triangular')

    >>> t = np.linspace(0,10,N)
    >>> plt.plot(t,x_1, label='half-sine')
    >>> plt.plot(t,x_2, label='triangular')
    >>> plt.show()
    """
    if not isinstance(n_start, int) or not isinstance(length, int) or not isinstance(N, int):
        raise ValueError("'N', 'n_start' and 'length' must be integers!")
    
    if  N < n_start + length:
        raise ValueError("'N' must be bigger or equal than 'n_start'+'length'!")

    pulse = np.zeros(N-n_start)

    if window != 'sawtooth':
        window_pulse = signal.windows.get_window(window, length)
        pulse[:length] = amplitude * window_pulse
    else: # until added to scipy.signal.windows module
        pulse[:length] = np.linspace(0, amplitude, length)

    pulse = np.pad(pulse, (n_start,0), mode='constant', constant_values=(0, 0))

    return pulse


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

    :param N: number of data points in returned signal
    :type N: int
    :param PSD: one-sided power spectral density
    :type PSD:  array
    :param fs: sampling period
    :type fs: int, float
    :param s_k: skewness of returned signal
    :type s_k: int, float
    :param k_u: kurtossis of returned signal
    :type k_u: int, float
    :param mean: mean value of returned signal
    :type mean: int, float
    :returns: stationary non-Gaussian realization of random process.

    References
    ----------
    [1] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
        2005
    [2] Steven R. Winterstein. Nonlinear vibration models for extremes and fatigue. ASCE Journal of Engineering
        Mechanics, 114:1772–1790, 1988.

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import signal_generation as sg

    >>> N = 1000 # number of data points of time signal
    >>> fs = 100 # sampling frequency [Hz]
    >>> t = np.arange(0,N)/fs # time vector
    >>> M = N // 2 + 1 # number of data points of frequency vector
    >>> f = np.arange(0, M, 1) * fs / N # frequency vector
    >>> f_min = 10 # PSD upper frequency limit  [Hz]
    >>> f_max = 20 # PSD lower frequency limit [Hz]

    >>> PSD = sg.get_psd(f, f_min, f_max) # one-sided flat-shaped PSD
    >>> x_gauss = sg.random_gaussian(N, PSD, fs) 
    >>> x_ngauss = sg.stationary_nongaussian_signal(N, PSD, fs, k_u = 5) 
    >>> plt.plot(t, x_gauss, label='gaussian')
    >>> plt.plot(t, x_ngauss, label='non-gaussian')
    >>> plt.xlabel(t [s])
    >>> plt.ylabel(x [unit])
    >>> plt.legend()
    >>> plt.show()
    """
    x = random_gaussian(N, PSD, fs) #gaussian random process

    h_4 = (np.sqrt(1 + 1.5*(k_u -3)) - 1)/18 #parameter h4 [2]
    h_3 = s_k/(6*(1 + 6*h_4)) ##parameter h3 [2]
    Κ = 1/np.sqrt(1 + 2*h_3**2 + 6*h_4**2) #parameter K [2]
    σ_x = np.std(x) #standard deviation of gaussian process
    nongaussian_signal = mean + Κ*(x/σ_x + h_3*(x/σ_x - 1) + h_4*((x/σ_x)**3 - 3*x/σ_x)) #[2]

    return nongaussian_signal


def _get_nonstationary_signal_psd(N, PSD, fs, PSD_modulating, p = 1, delta_m = 1):
    """
    Non-stationary non-Gaussian realization of random process.

    Non-stationarity random process is obtained by amplitude modulation of Gaussian random process[1].
    Gaussian random process is obtained with IFFT of amplitude spectra with random phase [2].
    Modulating signal is generated on PSD basis [3]. For internal use by `nonstationary_signal`.

    :param N: number of data points in returned signal
    :type N: int, float
    :param PSD: one-sided power spectral density of carrier signal
    :type PSD: array
    :param fs: sampling period
    :type fs: int, float
    :param PSD_modulating: one-sided power spectral density of modulating signal
    :type PSD_modulating: array
    :param p: exponent
    :type p: int, float
    :param delta_m: offset 
    :type delta_m: int, float
    :returns: nonstationary, stationary and modulating_signal

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
    
    return nonstationary_signal, stationary_signal, modulating_signal


def _get_nonstationary_signal_beta(N, PSD, fs, delta_n, alpha = 1, beta = 1):
    ''' 
    Non-stationary non-Gaussian realization of random process.

    Non-stationarity random process is obtained by amplitude modulation of Gaussian random process[1]. 
    Gaussian random process is obtained with IFFT of amplitude spectra with random phase [2]. Modulating
    signal is generated by cubic spline interpolation of points, based on beta distribution, defined by 
    parameters alpha and beta. For internal use by `nonstationary_signal`.

    :param N: number of data points in returned signal
    :type N: int, float
    :param PSD: one-sided power spectral density of carrier signal
    :type PSD: array
    :param fs: sampling period
    :type fs: int, float
    :param delta_n: sample step
    :type delta_n: int
    :param alpha: parameter of beta distribution
    :type alpha: float
    :param beta: parameter of beta distribution 
    :type beta: float
    :returns: nonstationary, stationary and modulating_signal
    
    References
    ----------
    [1] Frederic Kihm, Stephen A. Rizzi, N. S. Ferguson, and Andrew Halfpenny. Understanding how kurtosis is
        transferred from input acceleration to stress response and it’s influence on fatigue life. In Proceedings of the
        XI International Conference on Recent Advances in Structural Dynamics, Pisa, Italy, 07 2013.
    [2] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
        2005
    '''
    if (delta_n & (delta_n-1) == 0) and delta_n != 0 : # if step is power of 2
        delta_n = delta_n + 1

    t = np.arange(0, N) / fs # time vector
    t_beta = np.copy(t[::delta_n]) # time vector for modulating signal, with step delta_n
    t_beta[-1] = t[-1] # last data point in both time vectors are the same
    n = N//delta_n # number of data points for beta distribution
    points_beta = np.random.beta(alpha, beta, n + 1) 
    points_beta[-1] = points_beta[0] # first and last points are the same

    function_beta = CubicSpline(t_beta, points_beta, bc_type = 'periodic', extrapolate=None) 
    modulating_signal = function_beta(t) / np.std(function_beta(t)) # unit variance modulating signal 

    #shift to non-negative values
    if np.min(modulating_signal) < 0:
        modulating_signal += np.abs(np.min(modulating_signal))

    stationary_signal = random_gaussian(N, PSD, fs) # gaussian random process
    nonstationary_signal = stationary_signal * modulating_signal #non-stationary signal
    nonstationary_signal /= np.std(nonstationary_signal) # unit variance
    
    return nonstationary_signal, stationary_signal, modulating_signal


def nonstationary_signal(N, PSD, fs, k_u = 3, modulating_signal = ('PSD', None),
                         param1_list = None, param2_list = None, SQ = False):
    """
    Non-stationary non-Gaussian realization of random process.
    
    Non-stationarity random process is obtained by amplitude modulation of Gaussian random process[1]. 
    Gaussian random process is obtained with IFFT of amplitude spectra with random phase [2].
    Tuple modulating_signal selects the type of modulating signal: 'PSD' for random proces realization [3], where PSD_modulating
    is power spectrum density of modulating signal, and 'CSI' for cubic spline 
    interpolation [4], with sample step delta_n. The desired kurtosis k_u is obtained by iteration over lists param1_list 
    and param2_list (for 'PSD' p and delta_m are needed, for 'CSI' alpha and beta are needed).
    
    :param N: number of data points in returned signal
    :type N: (int, float)
    :param PSD: one-sided power spectral density of carrier signal
    :type PSD: array
    :param fs: sampling period
    :type fs: (int, float)
    :param k_u: desired kurtosis value of returned signal. Defaults to 3 (Gaussian random process).
    :type k_u: float
    :param modulating_signal: selects type of modulating signal and provides needed parameter.
    :type modulating_signal: tuple with name and parameter.
    :param param1_list: list of first parameter for modulating signal generation. p and alpha
    :type param1_list: list of floats
    :param param2_list: list of second parameter for modulating signal generation. delta_m and beta
    :type param2_list: list of floats
    :param SQ: If squeezing of signal [4] is required, set 'True'. Defaults to 'False'.
    :type SQ: boolean
    :returns: nonstationary, stationary and modulating_signal

    References
    ----------
    [1] Frederic Kihm, Stephen A. Rizzi, N. S. Ferguson, and Andrew Halfpenny. Understanding how kurtosis is
        transferred from input acceleration to stress response and it’s influence on fatigue life. In Proceedings of the
        XI International Conference on Recent Advances in Structural Dynamics, Pisa, Italy, 07 2013.
    [2] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
        2005
    [3] Arvid Trapp, Mafake James Makua, and PeterWolfsteiner. Fatigue assessment of amplitude-modulated nonstationary
        random vibration loading. Procedia Structural Integrity, 17:379—-386, 2019.
    [4] Lorenzo Capponi, Martin Česnik, Janko Slavič, Filippo Cianetti, and Miha Boltežar.  Non-stationarity index in 
        vibration fatigue: Theoretical and ex-perimental research.International Journal of Fatigue, 104:221–230, 2017.
        
    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import signal_generation as sg

    >>> N = 1000 # number of data points of time signal
    >>> fs = 100 # sampling frequency [Hz]
    >>> t = np.arange(0,N)/fs # time vector
    >>> M = N // 2 + 1 # number of data points of frequency vector
    >>> f = np.arange(0, M, 1) * fs / N # frequency vector
    >>> f_min = 10 # signals's PSD upper frequency limit  [Hz]
    >>> f_max = 20 # signals' PSD lower frequency limit [Hz]
    >>> f_min_mod = 1 # modulating signals's PSD upper frequency limit  [Hz]
    >>> f_max_mod = 2 # modulating signals's PSD lower frequency limit [Hz]

    #PSD of stationary signal
    >>> PSD = sg.get_psd(f, f_low = f_min, f_high = f_max) # one-sided flat-shaped PSD
    #PSD of modulating signal
    >>> PSD_modu = sg.get_psd(f, f_low = f_min_mod, f_high = f_max_mod) # one-sided flat-shaped PSD
    >>> k_u = 5
    #
    >>> x_nonstationary_1 = sg.nonstationary_signal(N,PSD,fs,k_u=k_u,modulating_signal=('PSD',PSD_modulating))

    #calculate kurtosis 
    >>> k_u_1 = sg.get_kurtosis(x_nonstationary_1)
    >>> print(f'desired kurtosis :{k_u:.3f}', actual kurtosis :{k_u_1:.3f}')

    #amplitude modulation parameters array with finer division
    >>> delta_m_list = np.arange(.1,2.1,.1) 
    >>> p_list = np.arange(.1,2.1,.1)
    >>> x_nonstationary_2 = sg.nonstationary_signal(N,PSD,fs,k_u=k_u,modulating_signal=('PSD',PSD_modulating),
                                                    param1_list=delta_m_list,param2_list=p_list)
    >>> k_u_2 = sg.get_kurtosis(x_nonstationary_2)
    >>> print(f'desired kurtosis :{k_u:.3f}', actual kurtosis :{k_u_2:.3f}')

    #define array of parameters alpha and beta
    >>> alpha_list = np.arange(1,7,1)
    >>> beta_list = np.arange(1,7,1)
    >>> x_nonstationary_3 = sg.nonstationary_signal(N,PSD,fs,k_u=10,modulating_signal=('CSI',delta_n),
                                                        param1_list=alpha_list,param2_list=beta_list)
    >>> k_u_3 = sg.get_kurtosis(x_nonstationary_3)
    >>> print(f'desired kurtosis :{k_u:.3f}', actual kurtosis :{k_u_3:.3f}')

    >>> plt.plot(t, x_nonstationary_2, label='PSD')
    >>> plt.plot(t, x_nonstationary_3, label='CSI)
    >>> plt.xlabel(t [s])
    >>> plt.ylabel(x [unit])
    >>> plt.legend()
    >>> plt.show()
    """
    #read type and parameter of modulating signal
    mod_signal_type, mod_sig_parameter = modulating_signal
    
    if mod_signal_type not in ['PSD', 'CSI']:
        raise ValueError('Valid options for `mod_signal_type` are `PSD` and `CSI` ')
    if mod_sig_parameter is None:
        raise ValueError('`mod_sig_parameter` must be specified!')
        
    #default param1/2 list, if not provided as argument of function        
    if param1_list is None:
        if mod_signal_type == 'PSD':
            param1_list = np.arange(.1,2,.1) # p
        else: #'CSI'
            param1_list = np.arange(1,7,1) # alpha

    if param2_list is None: 
        if mod_signal_type == 'PSD':
            param2_list = np.arange(0,1,.1) # delta_m
        else: #'CSI'
            param2_list = np.arange(1,7,1) #beta
        
    nonstationary_signals_tmp = {} # temporary signals dict
    delta_k_u_dict = {} # for difference of actual and targeted kurtosis   
    
    if SQ: #only if squeizzing is required
        stationary_signals_tmp = {} # temporary stationary signals dict
        modulation_signals_tmp = {} # temporary modulating signals dict

    for param1 in param1_list:     # p/alpha
        for param2 in param2_list:   # delta_m/beta

            if mod_signal_type == 'PSD':
                am_sig_tmp, sig_tmp, mod_tmp = _get_nonstationary_signal_psd(N, PSD, fs, 
                                                                             mod_sig_parameter, p = param1, delta_m = param2)
            elif mod_signal_type == 'CSI':
                am_sig_tmp, sig_tmp, mod_tmp = _get_nonstationary_signal_beta(N, PSD, fs, 
                                                                              mod_sig_parameter, alpha = param1, beta = param2)

            nonstationary_signals_tmp[f'param1={param1}, param2={param2}'] = am_sig_tmp
            k_u_tmp = moment(am_sig_tmp, 4)/(moment(am_sig_tmp, 2)**2)
            delta_k_u_dict[f'param1={param1}, param2={param2}'] = np.abs(k_u - k_u_tmp)
            
            if SQ: 
                stationary_signals_tmp[f'param1={param1}, param2={param2}'] = sig_tmp
                modulation_signals_tmp[f'param1={param1}, param2={param2}'] = mod_tmp
                       
    min_key = min(delta_k_u_dict, key=delta_k_u_dict.get) 
    
    if not SQ: 
        return nonstationary_signals_tmp[min_key]
    else: 
        return stationary_signals_tmp[min_key], modulation_signals_tmp[min_key]


def get_psd(f, f_low, f_high, variance = 1):
    '''
    One-sided flat-shaped power spectral density (PSD). 

    :param f: frequency vector [Hz]
    :type f: array
    :param f_low: lower frequency of PSD [Hz]
    :type f_low: float
    :param f_high: higher frequency of PSD [Hz]
    :type f_high: float
    :param variance: variance of random process, described by PSD [unit^2]
    :type variance: float
    :returns: one-sided flat-shaped PSD [unit^2/Hz]

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import signal_generation as sg

    >>> N = 1000 # number of data points of time signal
    >>> fs = 100 # sampling frequency [Hz]
    >>> t = np.arange(0,N)/fs # time vector
    >>> M = N // 2 + 1 # number of data points of frequency vector
    >>> f = np.arange(0, M, 1) * fs / N # frequency vector
    >>> f_min = 10 # PSD upper frequency limit  [Hz]
    >>> f_max = 20 # PSD lower frequency limit [Hz]

    >>> PSD = sg.get_psd(f, f_min, f_max) # one-sided flat-shaped PSD
    >>> plt.plot(f,PSD)
    >>> plt.xlabel(f [Hz])
    >>> plt.ylabel(PSD [unit^2/Hz])
    >>> plt.show()
    '''
    PSD = np.zeros(len(f)) 
    indx = np.logical_and(f>=f_low, f<=f_high) 
    PSD_width = f[indx][-1] - f[indx][0]
    PSD[indx] = variance/PSD_width # area under PSD is variance 
    return PSD


def get_kurtosis(signal):
    '''
    Kurtosis of signal. 

    :param signal: signal
    :type signal: array
    :returns: kurtosis
    '''
    μ_2 = moment(signal, 2)
    μ_4 = moment(signal, 4)
    k_u = μ_4/μ_2**2
    return k_u