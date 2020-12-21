import numpy as np
from scipy.stats import moment, beta
from scipy.interpolate import CubicSpline
from scipy import signal


def uniform_random(N, rg=None):
    """
    Uniform random distribution

    :param N: Number of points.
    :type N: int
    :param rg: Initialized Generator object
    :type rg: numpy.random._generator.Generator
    :returns: Random samples from a “uniform” distribution

    Example
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyExSi as es

    >>> N = 100
    >>> x = es.uniform_random(N=N)
    >>> plt.plot(x)
    >>> plt.show()
    """
    if rg == None:
        rg = np.random.default_rng()
    if isinstance(rg, np.random._generator.Generator):
        burst = rg.uniform(size=N) - 0.5
    else:
        raise ValueError(
            '`rg` must be initialized Generator object (numpy.random._generator.Generator)!'
        )

    return burst / np.max(np.abs(burst))


def normal_random(N, rg=None):
    """
    Normal random distribution.

    :param N: Number of points.
    :type N: int
    :param rg: Initialized Generator object
    :type rg: numpy.random._generator.Generator
    :returns: Random samples from the “standard normal” distribution

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyExSi as es

    >>> N = 100
    >>> x = es.uniform_random(N=N)
    >>> plt.plot(x)
    >>> plt.show()
    """
    if rg == None:
        rg = np.random.default_rng()
    if isinstance(rg, np.random._generator.Generator):
        burst = rg.standard_normal(size=N)
    else:
        raise ValueError(
            '`rg` must be initialized Generator object (numpy.random._generator.Generator)!'
        )

    return burst / np.max(np.abs(burst))


def pseudo_random(N, rg=None):
    """
    Pseudorandom distribution.

    Magnitudes are 1, phase is random.

    :param N: Number of points.
    :type N: int
    :param rg: Initialized Generator object
    :type rg: numpy.random._generator.Generator
    :returns: Random samples from the “standard normal” distribution

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyExSi as es

    >>> N = 100
    >>> x = es.pseudo_random(N=N)
    >>> plt.plot(x)
    >>> plt.show()
    """
    R = np.ones(N // 2 + 1, complex)

    if rg == None:
        rg = np.random.default_rng()
    if isinstance(rg, np.random._generator.Generator):
        R_prand = R * np.exp(1j * rg.uniform(size=len(R)) * 2 * np.pi)
    else:
        raise ValueError(
            '`rg` must be initialized Generator object (numpy.random._generator.Generator)!'
        )

    burst = np.fft.irfft(R_prand)
    return burst / np.max(np.abs(burst))


def burst_random(
    N,
    A=1.0,
    ratio=0.5,
    distribution='uniform',
    n_bursts=1,
    periodic_bursts=True,
    rg=None,
):
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
    :param rg: Initialized Generator object
    :type rg: numpy.random._generator.Generator
    :returns: Burst random signal time series.

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyExSi as es

    >>> N = 1000
    >>> amplitude = 5
    >>> x = es.burst_random(N, A=amplitude, ratio=0.1, distribution='normal', n_bursts=3)
    >>> plt.plot(x)
    >>> plt.show()
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
            br = uniform_random(N, rg=rg) * A
        elif distribution == 'normal':
            br = normal_random(N, rg=rg) * A
        elif distribution == 'pseudorandom':
            br = pseudo_random(N, rg=rg) * A
        else:
            raise ValueError(
                "Set `distribution` either to 'normal', 'uniform' or 'periodic'."
            )

        if ratio != 1.0:
            N_zero = int(np.floor(N * (1 - ratio)))
            br[-N_zero:] = 0.0

        bursts.append(br)
    bursts = np.asarray(bursts).flatten()

    if periodic_bursts:
        if n_bursts > 1:
            bursts = np.tile(bursts, n_bursts)

    return bursts


def sine_sweep(
    time, phi=0, freq_start=1, sweep_rate=None, freq_stop=None, mode='linear', phi_end=False
):
    """
    Generate a sine sweep signal time series.

    :param time: array of shape (N,), time vector.
    :param phi: float, initial phase of the sine signal in radians.
        Defaults to 0.
    :param freq_start: float, initial frequency in Hz.
    :param sweep_rate: float, the rate of sweep. In Hz/s for a linear sweep,
        in octaves/minute for a logarithmic sweep. If not given it is
        calculated from `time`, `freq_start` and `freq_stop`.
    :param freq_stop: float, final frequency in Hz.
    :param mode: 'linear' or 'logarithmic', type of sweep, optional.
        Defaults to 'linear'.
    :param phi_end: If True, return (`sweep_sine`, `phi_end`), where
       `phi_end` is the end phase which can be used as `phi` if this
       function is called for another sweep.
       Defaults to False.
    :returns: array of shape (N,), the generated sine sweep signal

    Example
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyExSi as es

    >>> t = np.linspace(0,10,1000)
    >>> x = es.sine_sweep(time=t, freq_start=0, freq_stop=5)
    >>> plt.plot(t, x)
    >>> plt.show()
    """
    if sweep_rate is None:
        if not freq_stop is None:
            T = time[-1] - time[0]
            sweep_rate = _sweep_rate(T, freq_start, freq_stop, mode)
        else:
            raise ValueError('`sweep_rate` is not given, please supply `freq_stop`.')
    if phi_end:
        # prepare time
        time_ = np.zeros(len(time) + 1)
        time_[: len(time)] = time
        time_[-1] = time[-1] + (time[-1] - time[-2])
    else:
        time_ = time

    if mode == 'linear':
        phase_t = 2 * np.pi * (sweep_rate * 0.5 * time_ ** 2 + freq_start * time_)
    elif mode == 'logarithmic':
        phase_t = (
            2
            * np.pi
            * 60
            * freq_start
            / (sweep_rate * np.log(2))
            * (2 ** (sweep_rate * time_ / 60) - 1)
        )
    else:
        raise ValueError(f"Invalid sweep mode `mode`='{mode}'.")

    s = np.sin(phase_t + phi)
    if phi_end:
        return s[:-1], phase_t[-1]
    else:
        return s


def _sweep_rate(T, freq_start, freq_stop, mode='linear'):
    """
    Calculate the sweep rate given the time difference, initial and end
    frequency values and sweep mode. For internal use by `sweep`.
    """
    if mode == 'linear':
        sweep_rate = (freq_stop - freq_start) / T  # Hz/s
    elif mode == 'logarithmic':
        sweep_rate = np.log((freq_stop / freq_start) ** (60 / T / np.log(2)))  # octaves/min
    else:
        raise ValueError('Invalid sweep mode `{mode}`.')
    return sweep_rate


def impulse(N, n_start=0, width=None, amplitude=1.0, window='sine'):
    """
    Impact impulse of the shape defined with the parameter window.

    :param N: Number of points in time signal.
    :type N: int
    :param width: Number of points for pulse width,  `None` results in width=N
    :type width: int
    :param amplitude: Amplitude of pulse.
    :type amplitude: float
    :param window:  The type of window to create. See scipy.signal.windows for more details.
    :type window: string, float, or tuple
    :returns: impact pulse.

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyExSi as es

    >>> N = 1000
    >>> n_start = 100
    >>> width = 200
    >>> amplitude = 3
    >>> x_1 = es.impulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window='triang')
    >>> x_2 = es.impulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window=('exponential',0,10))
    >>> t = np.linspace(0,10,N)
    >>> plt.plot(t,x_1, label='tringular')
    >>> plt.plot(t,x_2, label='exponential')
    >>> plt.legend()
    >>> plt.show()
    """
    if window == 'sine':
        window = 'cosine'
    if width is None:
        width = N
    if (
        not isinstance(n_start, int)
        or not isinstance(width, int)
        or not isinstance(N, int)
    ):
        raise ValueError('`N`, `n_start` and `width` must be integers!')

    if N < n_start + width:
        raise ValueError('`N` must be bigger than or equal to `n_start` + `length`!')

    pulse = np.zeros(N - n_start)

    if window != 'sawtooth':
        window_pulse = signal.windows.get_window(window, width)
        pulse[:width] = amplitude * window_pulse
    else:  # until sawtooth is added to scipy.signal.windows module
        pulse[:width] = np.linspace(0, amplitude, width)

    pulse = np.pad(pulse, (n_start, 0), mode='constant', constant_values=(0, 0))

    return pulse


def get_psd(freq, freq_lower, freq_upper, variance=1):
    """
    One-sided flat-shaped power spectral density (PSD).

    :param freq: Frequency vector [Hz]
    :type freq: array
    :param freq_lower: Lower frequency of PSD [Hz]
    :type freq_lower: float
    :param freq_upper: Upper frequency of PSD [Hz]
    :type freq_upper: float
    :param variance: Variance of random process, described by PSD [unit^2]
    :type variance: float
    :returns: one-sided flat-shaped PSD [unit^2/Hz]

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyExSi as es

    >>> N = 1000 # number of data points of time signal
    >>> fs = 100 # sampling frequency [Hz]
    >>> t = np.arange(0,N)/fs # time vector
    >>> M = N // 2 + 1 # number of data points of frequency vector
    >>> freq = np.arange(0, M, 1) * fs / N # frequency vector
    >>> freq_lower = 10 # PSD lower frequency limit  [Hz]
    >>> freq_upper = 20 # PSD upper frequency limit [Hz]

    >>> PSD = es.get_psd(freq, freq_lower, freq_upper) # one-sided flat-shaped PSD
    >>> plt.plot(freq,PSD)
    >>> plt.xlabel(f [Hz])
    >>> plt.ylabel(PSD [unit^2/Hz])
    >>> plt.show()
    """
    PSD = np.zeros(len(freq))
    indx = np.logical_and(freq >= freq_lower, freq <= freq_upper)
    PSD_width = freq[indx][-1] - freq[indx][0]
    PSD[indx] = variance / PSD_width  # area under PSD is variance
    return PSD


def random_gaussian(N, PSD, fs, rg=None):
    """
    Stationary Gaussian realization of random process, characterized by PSD.

    Random process is obtained with IFFT of amplitude spectra with random phase [1]. Area under PSD curve represents variance of random process.

    :param N: Number of points.
    :type N: int
    :param PSD: one-sided power spectral density [unit^2].
    :type PSD: array
    :param fs: sampling frequency [Hz].
    :type fs: int,float
    :param rg: Initialized Generator object
    :type rg: numpy.random._generator.Generator
    :returns: stationary Gaussian realization of random process

    References
    ----------
    [1] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis.
    Dover Publications, 2005

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyExSi as es

    >>> N = 1000 # number of data points of time signal
    >>> fs = 100 # sampling frequency [Hz]
    >>> t = np.arange(0,N)/fs # time vector
    >>> M = N // 2 + 1 # number of data points in frequency vector
    >>> freq = np.arange(0, M, 1) * fs / N # frequency vector
    >>> freq_lower = 10 # PSD lower frequency limit  [Hz]
    >>> freq_upper = 20 # PSD upper frequency limit [Hz]

    >>> PSD = es.get_psd(freq, freq_lower, freq_upper) # one-sided flat-shaped PSD
    >>> x = es.random_gaussian(N, PSD, fs)
    >>> plt.plot(t,x)
    >>> plt.xlabel(t [s])
    >>> plt.ylabel(x [unit])
    >>> plt.show()
    """
    ampl_spectra = np.sqrt(PSD * N * fs / 2)  # amplitude spectra

    if rg == None:
        rg = np.random.default_rng()
    if isinstance(rg, np.random._generator.Generator):
        ampl_spectra_random = ampl_spectra * np.exp(
            1j * rg.uniform(0, 1, len(PSD)) * 2 * np.pi
        )  # amplitude spectra, random phase
    else:
        raise ValueError(
            '`rg` must be initialized Generator object (numpy.random._generator.Generator)!'
        )

    burst = np.fft.irfft(ampl_spectra_random)  # time signal
    return burst


def stationary_nongaussian_signal(N, PSD, fs, s_k=0, k_u=3, mean=0, rg=None):
    """
    Stationary non-Gaussian realization of random process.

    Random process is obtained with IFFT of amplitude spectra with random phase [1]. Non-Gaussianity is obtained by Winterstein polynomials [2].

    :param N: number of data points in returned signal
    :type N: int
    :param PSD: one-sided power spectral density
    :type PSD:  array
    :param fs: sampling frequency
    :type fs: int, float
    :param s_k: skewness of returned signal
    :type s_k: int, float
    :param k_u: kurtossis of returned signal
    :type k_u: int, float
    :param mean: mean value of returned signal
    :type mean: int, float
    :param rg: Initialized Generator object
    :type rg: numpy.random._generator.Generator
    :returns: stationary non-Gaussian realization of random process.

    References
    ----------
    [1] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet
    Analysis. Dover Publications, 2005

    [2] Steven R. Winterstein. Nonlinear vibration models for extremes and
    fatigue. ASCE Journal of Engineering Mechanics, 114:1772–1790, 1988.

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyExSi as es

    >>> N = 1000 # number of data points of time signal
    >>> fs = 100 # sampling frequency [Hz]
    >>> t = np.arange(0,N)/fs # time vector
    >>> M = N // 2 + 1 # number of data points of frequency vector
    >>> freq = np.arange(0, M, 1) * fs / N # frequency vector
    >>> freq_lower = 10 # PSD lower frequency limit  [Hz]
    >>> freq_upper = 20 # PSD upper frequency limit [Hz]

    >>> PSD = es.get_psd(freq, freq_lower, freq_upper) # one-sided flat-shaped PSD
    >>> x_gauss = es.random_gaussian(N, PSD, fs)
    >>> x_ngauss = es.stationary_nongaussian_signal(N, PSD, fs, k_u = 5)
    >>> plt.plot(t, x_gauss, label='gaussian')
    >>> plt.plot(t, x_ngauss, label='non-gaussian')
    >>> plt.xlabel(t [s])
    >>> plt.ylabel(x [unit])
    >>> plt.legend()
    >>> plt.show()
    """
    x = random_gaussian(N, PSD, fs, rg=rg)  # gaussian random process

    h_4 = (np.sqrt(1 + 1.5 * (k_u - 3)) - 1) / 18  # parameter h4 [2]
    h_3 = s_k / (6 * (1 + 6 * h_4))  ##parameter h3 [2]
    Κ = 1 / np.sqrt(1 + 2 * h_3 ** 2 + 6 * h_4 ** 2)  # parameter K [2]
    sigma_x = np.std(x)  # standard deviation of gaussian process
    nongaussian_signal = mean + Κ * (
        x / sigma_x
        + h_3 * (x / sigma_x - 1)
        + h_4 * ((x / sigma_x) ** 3 - 3 * x / sigma_x)
    )  # [2]

    return nongaussian_signal


def _get_nonstationary_signal_psd(N, PSD, fs, PSD_modulating, p=1, delta_m=1, rg=None):
    """
    Non-stationary non-Gaussian realization of random process.

    Non-stationarity random process is obtained by amplitude modulation of Gaussian random process[1].
    Gaussian random process is obtained with IFFT of amplitude spectra with random phase [2].
    Modulating signal is generated on PSD basis [3]. For internal use by `nonstationary_signal`.

    :param N: number of data points in returned signal
    :type N: int, float
    :param PSD: one-sided power spectral density of carrier signal
    :type PSD: array
    :param fs: sampling frequency
    :type fs: int, float
    :param PSD_modulating: one-sided power spectral density of modulating signal
    :type PSD_modulating: array
    :param p: exponent
    :type p: int, float
    :param delta_m: offset
    :type delta_m: int, float
    :param rg: Initialized Generator object
    :type rg: numpy.random._generator.Generator
    :returns: nonstationary, stationary and modulating_signal

    References
    ----------
    [1] Frederic Kihm, Stephen A. Rizzi, N. S. Ferguson, and Andrew Halfpenny.
    Understanding how kurtosis is transferred from input acceleration to stress
    response and it’s influence on fatigue life. In Proceedings of the XI
    International Conference on Recent Advances in Structural Dynamics, Pisa,
    Italy, 07 2013.

    [2] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet
    Analysis. Dover Publications, 2005

    [3] Arvid Trapp, Mafake James Makua, and Peter Wolfsteiner. Fatigue
    assessment of amplitude-modulated nonstationary random vibration loading.
    Procedia Structural Integrity, 17:379—-386, 2019.

    """
    stationary_signal = random_gaussian(
        N, PSD, fs, rg=rg
    )  # gaussian random process, carrier
    modulating_signal = random_gaussian(
        N, PSD_modulating, fs, rg=rg
    )  # gaussian random process,  modulating signal

    nonstationary_signal = stationary_signal * (
        np.abs(modulating_signal) ** p + delta_m
    )  # [3]
    nonstationary_signal = nonstationary_signal / np.std(
        nonstationary_signal
    )  # non-stationary signal

    return nonstationary_signal, stationary_signal, modulating_signal


def _get_nonstationary_signal_beta(N, PSD, fs, delta_n, alpha=1, beta=1, rg=None):
    """
    Non-stationary non-Gaussian realization of random process.

    Non-stationarity random process is obtained by amplitude modulation of Gaussian random process[1].
    Gaussian random process is obtained with IFFT of amplitude spectra with random phase [2]. Modulating
    signal is generated by cubic spline interpolation of points, based on beta distribution, defined by
    parameters alpha and beta. For internal use by `nonstationary_signal`.

    :param N: Number of data points in returned signal
    :type N: int, float
    :param PSD: One-sided power spectral density of carrier signal
    :type PSD: array
    :param fs: sampling frequency
    :type fs: int, float
    :param delta_n: Distance beetwen consecutive beta distributed points. Smaller delta_n corresponds to hihger modulation frequency.
    :type delta_n: int
    :param alpha: Parameter of beta distribution
    :type alpha: float
    :param beta: Parameter of beta distribution
    :type beta: float
    :param rg: Initialized Generator object
    :type rg: numpy.random._generator.Generator
    :returns: nonstationary, stationary and modulating_signal

    References
    ----------
    [1] Frederic Kihm, Stephen A. Rizzi, N. S. Ferguson, and Andrew Halfpenny.
    Understanding how kurtosis is transferred from input acceleration to
    stress response and it’s influence on fatigue life. In Proceedings of the
    XI International Conference on Recent Advances in Structural Dynamics, Pisa,
    Italy, 07 2013.

    [2] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet
    Analysis. Dover Publications, 2005
    """
    stationary_signal = random_gaussian(N, PSD, fs, rg=rg)  # gaussian random process

    t = np.arange(0, N) / fs  # time vector
    n = N // delta_n  # number of time intervals for beta distribution points
    t_beta = np.copy(
        t[: n * delta_n + 1 : delta_n]
    )  # time vector for modulating signal, with step delta_n
    t_beta = np.append(t_beta, t[-1])
    if N % delta_n != 0:
        n += 1
    t_beta[-1] = t[-1]

    if rg == None:
        rg = np.random.default_rng()
    if isinstance(rg, np.random._generator.Generator):
        points_beta = rg.beta(alpha, beta, n + 1)
        points_beta[-1] = points_beta[0]  # first and last points are the same
    else:
        raise ValueError(
            "rg' must be initialized Generator object (numpy.random._generator.Generator)!"
        )

    points_beta[-1] = points_beta[0]  # first and last points are the same
    function_beta = CubicSpline(
        t_beta, points_beta, bc_type='periodic', extrapolate=None
    )
    modulating_signal = function_beta(t) / np.std(
        function_beta(t)
    )  # unit variance modulating signal

    # shift to non-negative values
    if np.min(modulating_signal) < 0:
        modulating_signal += np.abs(np.min(modulating_signal))

    nonstationary_signal = (
        stationary_signal * modulating_signal[: len(stationary_signal)]
    )  # non-stationary signal
    nonstationary_signal /= np.std(nonstationary_signal)  # unit variance

    return nonstationary_signal, stationary_signal, modulating_signal


def nonstationary_signal(
    N,
    PSD,
    fs,
    k_u=3,
    modulating_signal=('PSD', None),
    param1_list=None,
    param2_list=None,
    seed=None,
    SQ=False,
):
    """
    Non-stationary non-Gaussian realization of random process.

    Non-stationarity random process is obtained by amplitude modulation of
    Gaussian random process[1].  Gaussian random process is obtained with IFFT
    of amplitude spectra with random phase [2]. Tuple modulating_signal selects
    the type of modulating signal: 'PSD' for random process realization [3],
    where PSD_modulating is power spectrum density of modulating signal, and
    'CSI' for cubic spline interpolation [4,5], with sample step delta_n.
    The desired kurtosis k_u is obtained by iteration over lists param1_list
    and param2_list (for 'PSD' p and delta_m are needed, for 'CSI' alpha and
    beta are needed).

    :param N: Number of data points in returned signal
    :type N: {int, float}
    :param PSD: One-sided power spectral density of carrier signal
    :type PSD: array
    :param fs: sampling frequency
    :type fs: {int, float}
    :param k_u: Desired kurtosis value of returned signal. Defaults to 3 (Gaussian random process).
    :type k_u: float
    :param modulating_signal: Delects type of modulating signal and provides needed parameter.
    :type modulating_signal: tuple with name and parameter.
    :param param1_list: List of first parameter for modulating signal generation. Contains parameters p or alpha
    :type param1_list: list of floats
    :param param2_list: List of second parameter for modulating signal generation. Contains parameters delta_m or beta
    :type param2_list: list of floats
    :param seed: A seed to initialize the BitGenerator. For details, see numpy.random.default_rng()
    :type seed: {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
    :param SQ: If squeezing of signal [4] is required, set 'True'. Defaults to 'False'
    :type SQ: boolean
    :returns: nonstationary signal. Optionally, stationary and modulating_signal are returned as well.

    References
    ----------
    [1] Frederic Kihm, Stephen A. Rizzi, N. S. Ferguson, and Andrew Halfpenny.
    Understanding how kurtosis is transferred from input acceleration to stress
    response and it’s influence on fatigue life. In Proceedings of the XI
    International Conference on Recent Advances in Structural Dynamics, Pisa,
    Italy, 07 2013.

    [2] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet
    Analysis. Dover Publications, 2005

    [3] Arvid Trapp, Mafake James Makua, and Peter Wolfsteiner. Fatigue
    assessment of amplitude-modulated nonstationary random vibration loading.
    Procedia Structural Integrity, 17:379—-386, 2019.

    [4] Lorenzo Capponi, Martin Česnik, Janko Slavič, Filippo Cianetti, and
    Miha Boltežar.  Non-stationarity index in vibration fatigue: Theoretical
    and ex-perimental research.International Journal of Fatigue, 104:221–230,
    2017.

    [5] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
    Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
    – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020

    Example
    --------
    >>> import numy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyExSi as es

    >>> N = 1000 # number of data points of time signal
    >>> fs = 100 # sampling frequency [Hz]
    >>> t = np.arange(0,N)/fs # time vector
    >>> M = N // 2 + 1 # number of data points of frequency vector
    >>> freq = np.arange(0, M, 1) * fs / N # frequency vector
    >>> freq_lower = 10 # PSD lower frequency limit  [Hz]
    >>> freq_upper = 20 # PSD upper frequency limit [Hz]
    >>> freq_lower_mod = 1 # modulating signals's PSD lower frequency limit  [Hz]
    >>> freq_upper_mod = 2 # modulating signals's PSD upper frequency limit [Hz]

    PSD of stationary and modulating signal

    >>> PSD = es.get_psd(freq, freq_lower, freq_upper) # one-sided flat-shaped PSD
    >>> PSD_modulating = es.get_psd(freq, freq_lower_mod, freq_upper_mod) # one-sided flat-shaped PSD

    Specify kurtosis and return non-stationary signal

    >>> k_u = 5
    >>> x_nonstationary_1 = es.nonstationary_signal(N,PSD,fs,k_u=k_u,modulating_signal=('PSD',PSD_modulating))

    Calculate kurtosis

    >>> k_u_1 = es.get_kurtosis(x_nonstationary_1)
    >>> print(f'desired kurtosis :{k_u:.3f}', actual kurtosis :{k_u_1:.3f}')

    Refined array with amplitude modulation parameters

    >>> delta_m_list = np.arange(.1,2.1,.1)
    >>> p_list = np.arange(.1,2.1,.1)
    >>> x_nonstationary_2 = es.nonstationary_signal(N,PSD,fs,k_u=k_u,modulating_signal=('PSD',PSD_modulating),
                                                    param1_list=delta_m_list,param2_list=p_list)
    >>> k_u_2 = es.get_kurtosis(x_nonstationary_2)
    >>> print(f'desired kurtosis :{k_u:.3f}', actual kurtosis :{k_u_2:.3f}')

    Define array of parameters alpha and beta

    >>> alpha_list = np.arange(1,4,.5)
    >>> beta_list = np.arange(1,4,.5)
    >>> x_nonstationary_3 = es.nonstationary_signal(N,PSD,fs,k_u=10,modulating_signal=('CSI',delta_n),
                                                        param1_list=alpha_list,param2_list=beta_list)
    >>> k_u_3 = es.get_kurtosis(x_nonstationary_3)
    >>> print(f'desired kurtosis :{k_u:.3f}', actual kurtosis :{k_u_3:.3f}')

    >>> plt.plot(t, x_nonstationary_2, label='PSD')
    >>> plt.plot(t, x_nonstationary_3, label='CSI)
    >>> plt.xlabel(t [s])
    >>> plt.ylabel(x [unit])
    >>> plt.legend()
    >>> plt.show()
    """
    # read type and parameter of modulating signal
    mod_signal_type, mod_sig_parameter = modulating_signal

    # default param1/2 list, if not provided as function argument
    if param1_list is None:
        if mod_signal_type == 'PSD':
            param1_list = np.arange(0.1, 2, 0.1)  # p
        else:  #'CSI'
            param1_list = np.arange(1, 10, 0.5)  # alpha

    if param2_list is None:
        if mod_signal_type == 'PSD':
            param2_list = np.arange(0, 1, 0.1)  # delta_m
        else:  #'CSI'
            param2_list = np.arange(1, 10, 0.5)  # beta

    nonstationary_signals_tmp = {}  # temporary signals dict
    delta_k_u_dict = {}  # for difference of actual and targeted kurtosis

    if SQ:  # only if squeizzing is required
        stationary_signals_tmp = {}  # temporary stationary signals dict
        modulation_signals_tmp = {}  # temporary modulating signals dict

    for param1 in param1_list:  # p/alpha
        for param2 in param2_list:  # delta_m/beta
            if seed == None:
                rg = None
            elif isinstance(seed, int):
                rg = np.random.default_rng(seed)
            else:
                raise ValueError(
                    '`seed` must be of type {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}!'
                )

            if mod_signal_type == 'PSD':
                am_sig_tmp, sig_tmp, mod_tmp = _get_nonstationary_signal_psd(
                    N, PSD, fs, mod_sig_parameter, p=param1, delta_m=param2, rg=rg
                )
            elif mod_signal_type == 'CSI':
                am_sig_tmp, sig_tmp, mod_tmp = _get_nonstationary_signal_beta(
                    N, PSD, fs, mod_sig_parameter, alpha=param1, beta=param2, rg=rg
                )
            else:
                raise ValueError(
                    'Valid options for `mod_signal_type` are `PSD` and `CSI` '
                )

            nonstationary_signals_tmp[f'param1={param1}, param2={param2}'] = am_sig_tmp
            k_u_tmp = moment(am_sig_tmp, 4) / (moment(am_sig_tmp, 2) ** 2)
            delta_k_u_dict[f'param1={param1}, param2={param2}'] = np.abs(k_u - k_u_tmp)

            if SQ:
                stationary_signals_tmp[f'param1={param1}, param2={param2}'] = sig_tmp
                modulation_signals_tmp[f'param1={param1}, param2={param2}'] = mod_tmp

    min_key = min(delta_k_u_dict, key=delta_k_u_dict.get)

    if not SQ:
        return nonstationary_signals_tmp[min_key]
    else:
        return stationary_signals_tmp[min_key], modulation_signals_tmp[min_key]


def get_kurtosis(signal):
    """
    Kurtosis of signal.

    :param signal: input signal.
    :type signal: array
    :returns: kurtosis
    """
    μ_2 = moment(signal, 2)
    μ_4 = moment(signal, 4)
    k_u = μ_4 / μ_2 ** 2
    return k_u


if __name__ == "__main__":
    time = np.linspace(0, 1, 100)
    a = sine_sweep(time=time, sweep_rate=1)
    print(a)
