import numpy as np

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