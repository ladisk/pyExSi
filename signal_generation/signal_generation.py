import numpy as np
import scipy as sp


def get_psd(f, f_low, f_high, variance = 1):
    ''' Returns one-sided flat-shaped power spectral density (PSD). 


        :params:
        f - frequency vector
        f_low - lower frequency of PSD
        f_high - higher frequency of PSD 
        variance - variance of random process, described by PSD

        :returns:
        PSD - one-sided flat-shaped PSD vith specified variance
    '''
    
    PSD = np.zeros(len(f)) 
    indx = np.logical_and(f>=f_low, f<=f_high) 
    PSD_width = f[indx][-1] - f[indx][0]
    PSD[indx] = variance/PSD_width # area under PSD is variance 
    return PSD


def get_gaussian_signal(PSD, fs, N):
    ''' Returns stationary Gaussian realization of random process, characterized by PSD. 
        Random process is obtained with IFFT of amplitude spectra with random phase [1].


        :params:
        PSD - one-sided PSD 
        fs - sampling period
        N - number of data points in returned signal

        :returns:
        random_signal - stationary Gaussian realization of random process, characterized by PSD.

        References
        ----------
        [1] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
            2005
    '''
    
    A = np.sqrt(PSD*N*fs/2)  # amplitude spectra
    M = int(N/2 + 1) # number of data points in frequency vector
    phi = np.random.uniform(-np.pi, np.pi, M) #random phase
    A_rnd = A*np.exp(1j*phi) # amplitude spectra with random phase
    random_signal = np.fft.irfft(A_rnd) # random proces as IFFT of amplitude spectra with random phase
    return random_signal



def get_stationary_nongaussian_signal(PSD, fs, N, s_k = 0, k_u = 3, mean = 0, variance = 1):
    ''' Returns stationary non-Gaussian realization of random process, characterized by PSD.
        Random process is obtained with IFFT of amplitude spectra with random phase [1]. Non-Gaussianity
        is obtained by Winterstein polynomials [2]. 


        :params:
        PSD - one-sided PSD
        fs - sampling period
        N - number of data points in returned signal
        s_k - skewness of returned signal
        k_u - kurtossis of returned signal
        mean - mean value of returned signal
        variance - variance of returned signal

        :returns:
        nongaussian_signal - stationary non-Gaussian realization of random process.

        References
        ----------
        [1] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
            2005
        [2] Steven R. Winterstein. Nonlinear vibration models for extremes and fatigue. ASCE Journal of Engineering
            Mechanics, 114:1772–1790, 1988.

    '''
    
    x = get_gaussian_signal(PSD, fs, N) #gaussian random process
    h_4 = (np.sqrt(1 + 1.5*(k_u -3)) - 1)/18 #parameter h4 [2]
    h_3 = s_k/(6*(1 + 6*h_4)) ##parameter h3 [2]
    Κ = 1/np.sqrt(1 + 2*h_3**2 + 6*h_4**2) #parameter K [2]
    σ_x = np.std(x) #standard deviation of gaussian process
    nongaussian_signal = mean + variance*Κ*(x/σ_x + h_3*(x/σ_x - 1) + h_4*((x/σ_x)**3 - 3*x/σ_x)) #[2]
    return nongaussian_signal



def _get_nonstationary_signal_psd(PSD, PSD_modulating, fs, N, variance = 1, delta_m = 1, p = 1):
    ''' Returns non-stationary non-Gaussian realization of random process, characterized by PSD.
        Non-stationarity random process is obtained by amplitude modulation of Gaussian random process[1]. 
        Gaussian random process is obtained with IFFT of amplitude spectra with random phase [2]. Modulating
        signal is generated on PSD basis [3].


        :params:
        PSD - one-sided PSD
        PSD_modulating - one-sided PSD of modulating signal
        fs - sampling period
        N - number of data points in returned signal
        variance - variance of returned signal
        delta_m - offset 
        p - exponent
        
        :returns:
        nonstationary_signal - stationary non-Gaussian realization of random process.

        References
        ----------
        [1] Frederic Kihm, Stephen A. Rizzi, N. S. Ferguson, and Andrew Halfpenny. Understanding how kurtosis is
            transferred from input acceleration to stress response and it’s influence on fatigue life. In Proceedings of the
            XI International Conference on Recent Advances in Structural Dynamics, Pisa, Italy, 07 2013.
        [2] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
            2005
        [3] Arvid Trapp, Mafake James Makua, and PeterWolfsteiner. Fatigue assessment of amplitude-modulated nonstationary
            random vibration loading. Procedia Structural Integrity, 17:379—-386, 2019.

    '''
    
    stationary_signal = get_gaussian_signal(PSD, fs, N) #gaussian random process
    modulating_signal = get_gaussian_signal(PSD_modulating, fs, N) ##gaussian random process, as modulating signal
    
    nonstationary_signal = stationary_signal*(np.abs(modulating_signal)**p + delta_m) #[3]
    nonstationary_signal = nonstationary_signal/np.std(nonstationary_signal)*np.sqrt(variance) #non-stationary signal of specified variance
    return nonstationary_signal


def get_nonstationary_signal_psd(PSD, PSD_modulating, fs, N, delta_m_list, p_list, k_u = 3, variance = 1):
    ''' Returns non-stationary non-Gaussian realization of random process, characterized by PSD.
        Non-stationarity random process is obtained by amplitude modulation of Gaussian random process[1]. 
        Gaussian random process is obtained with IFFT of amplitude spectra with random phase [2]. Modulating
        signal is generated on PSD basis [3].


        :params:
        PSD - one-sided PSD
        PSD_modulating - one-sided PSD of modulating signal
        fs - sampling period
        N - number of data points in returned signal
        variance - variance of returned signal
        k_u - targeted kurtossis of returned signal


        :returns:
        nonstationary_signal - stationary non-Gaussian realization of random process.

        References
        ----------
        [1] Frederic Kihm, Stephen A. Rizzi, N. S. Ferguson, and Andrew Halfpenny. Understanding how kurtosis is
            transferred from input acceleration to stress response and it’s influence on fatigue life. In Proceedings of the
            XI International Conference on Recent Advances in Structural Dynamics, Pisa, Italy, 07 2013.
        [2] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
            2005
        [3] Arvid Trapp, Mafake James Makua, and PeterWolfsteiner. Fatigue assessment of amplitude-modulated nonstationary
            random vibration loading. Procedia Structural Integrity, 17:379—-386, 2019.

    '''
    
    nonstationary_signals_tmp = {} # temporary signals dict
    delta_k_u_dict = {} # difference of actual and targeted kurtosis     


    for p in p_list:     
        for delta_m in delta_m_list: 
            sig_tmp = _get_nonstationary_signal_psd(PSD, PSD_modulating, fs, N, variance = variance, delta_m=delta_m, p = p)
            nonstationary_signals_tmp[f'ind_p={p}, ind_m={delta_m}'] = sig_tmp
            k_u_tmp = sp.stats.moment(sig_tmp, 4)/(sp.stats.moment(sig_tmp, 2)**2)
            delta_k_u_dict[f'ind_p={p}, ind_m={delta_m}'] = np.abs(k_u - k_u_tmp)
            
    min_key = min(delta_k_u_dict, key=delta_k_u_dict.get) 
    
    return nonstationary_signals_tmp[min_key]



def _get_nonstationary_signal_beta(PSD, fs, N, delta_n, variance = 1, alpha = 1, beta = 1, SQ=False):
    ''' Returns non-stationary non-Gaussian realization of random process, characterized by PSD.
        Non-stationarity random process is obtained by amplitude modulation of Gaussian random process[1]. 
        Gaussian random process is obtained with IFFT of amplitude spectra with random phase [2]. Modulating
        signal is generated by cubic spline interpolation of points, based on beta distribution, defined by 
        parameters alpha and beta.


        :params:
        PSD - one-sided PSD
        fs - sampling period
        N - number of data points in returned signal
        delta_n - sample step for 
        alpha, beta - parameters of beta distribution
        variance - variance of returned signal
        SQ - If true, modulating signal is also returned

        :returns:
        nonstationary_signal - if SQ=False, returns stationary non-Gaussian realization of random process.
        modulating signal - if SQ=True, returns stationary gaussian signal and modulation signal

        References
        ----------
        [1] Frederic Kihm, Stephen A. Rizzi, N. S. Ferguson, and Andrew Halfpenny. Understanding how kurtosis is
            transferred from input acceleration to stress response and it’s influence on fatigue life. In Proceedings of the
            XI International Conference on Recent Advances in Structural Dynamics, Pisa, Italy, 07 2013.
        [2] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
            2005

    '''
    
    if (delta_n & (delta_n-1) == 0) and delta_n != 0 : #if power of 2
        delta_n = delta_n + 1

    t = np.arange(0,N)/fs #time vector
    t_beta = t[::delta_n] #time vector for modulating signal, with step delta_n
    t_beta[-1] = t[-1] #last data point in both time vectors are the same

    n = int(N/delta_n) #number of data points for beta distribution
    points_beta = np.random.beta(alpha,beta,n+1) 
    points_beta[-1] = points_beta[0] #first and last points are the same

    function_beta = sp.interpolate.CubicSpline(t_beta, points_beta, extrapolate = None) 
    modulating_signal = function_beta(t)/np.std(function_beta(t)) #evaluated modulating signal, normalized to unit standard deviation

    #shift to non-negative values
    if np.min(modulating_signal) < 0:
        modulating_signal += np.abs(np.min(modulating_signal))


    x_stationary = get_gaussian_signal(PSD, fs, N) #gaussian random process
    nonstationary_signal = x_stationary*modulating_signal/np.std(x_stationary*modulating_signal)*np.sqrt(variance) #non-stationary signal
    
    
    if not SQ: 
        return nonstationary_signal
    else: 
        return x_stationary, modulating_signal



def get_nonstationary_signal_beta(PSD, fs, N, delta_n, alpha_list, beta_list, k_u = 3, variance = 1, SQ=False):
    ''' Returns non-stationary non-Gaussian realization of random process, characterized by PSD.
        Non-stationarity random process is obtained by amplitude modulation of Gaussian random process[1]. 
        Gaussian random process is obtained with IFFT of amplitude spectra with random phase [2]. Modulating
        signal is generated by cubic spline interpolation of points, based on beta distribution, defined by 
        parameters alpha and beta.


        :params:
        PSD - one-sided PSD
        fs - sampling period
        N - number of data points in returned signal
        delta_n - sample step for 
        alpha, beta - parameters of beta distribution
        variance - variance of returned signal
        SQ - If true, modulating signal is also returned

        :returns:
        nonstationary_signal - if SQ=False, returns stationary non-Gaussian realization of random process.
        stationary signal, modulating signal - if SQ=True, returns stationary gaussian signal and modulation signal

        References
        ----------
        [1] Frederic Kihm, Stephen A. Rizzi, N. S. Ferguson, and Andrew Halfpenny. Understanding how kurtosis is
            transferred from input acceleration to stress response and it’s influence on fatigue life. In Proceedings of the
            XI International Conference on Recent Advances in Structural Dynamics, Pisa, Italy, 07 2013.
        [2] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis. Dover Publications,
            2005

    '''
    
    nonstationary_signals_tmp = {} # temporary signals dict
    
    if SQ: #only if squeizzing is needed later
        stationary_signals_tmp = {}
        modulation_signals_tmp = {}
        
    delta_k_u_dict = {} # difference of actual and targeted kurtosis     


    for alpha in alpha_list:     
        for beta in beta_list: 
            sig_nst_tmp = _get_nonstationary_signal_beta(PSD, fs, N, delta_n, alpha = alpha, beta=beta, variance = variance, SQ=False)
            nonstationary_signals_tmp[f'alpha={alpha}, beta={beta}'] = sig_nst_tmp
            k_u_tmp = sp.stats.moment(sig_nst_tmp, 4)/(sp.stats.moment(sig_nst_tmp, 2)**2)
            delta_k_u_dict[f'alpha={alpha}, beta={beta}'] = np.abs(k_u - k_u_tmp)
            
            if SQ:
                sig_tmp, mod_tmp = _get_nonstationary_signal_beta(PSD, fs, N, delta_n, alpha = alpha, beta=beta, variance = variance, SQ = SQ)
                stationary_signals_tmp[f'alpha={alpha}, beta={beta}'] = sig_tmp
                modulation_signals_tmp[f'alpha={alpha}, beta={beta}'] = mod_tmp
                       
    min_key = min(delta_k_u_dict, key=delta_k_u_dict.get) #dict key with minimal deviation from desired kurtosis
    
    if not SQ: 
        return nonstationary_signals_tmp[min_key]
    else: 
        return stationary_signals_tmp[min_key], modulation_signals_tmp[min_key]