signal_generation.py
---------------------------------------------
-Uniform random distribution

-Normal random distribution

-Pseudorandom distribution

-Zero-mean burst random

-Sweep

-Random signals, defined by power spectral density (PSD):stationary Gaussian, stationary non-Gaussian and non-stationary non-Gaussian random process



Simple example
---------------

A simple example on how to generate random signals on PSD basis:

.. code-block:: python

    import signal_generation as sg
    import numpy as np


    N = 2**16 # number of data points of time signal
    fs = 1024 # sampling frequency [Hz]
    t = np.arange(0,N)/fs # time vector

    # define frequency vector and one-sided flat-shaped PSD
    M = N//2 + 1 # number of data points of frequency vector
    f = np.arange(0, M, 1) * fs / N # frequency vector
    f_min = 50 # PSD upper frequency limit  [Hz]
    f_max = 100 # PSD lower frequency limit [Hz]
    PSD = sg.get_psd(f, f_min, f_max) # one-sided flat-shaped PSD

    #get gaussian stationary signal
    gausian_signal = sg.get_gaussian_signal(PSD, fs, N)

    #get non-gaussian non-stationary signal, with kurtosis k_u=10
    #amplitude modulation, modulating signal defined by PSD
    PSD_modulating = sg.get_psd(f, f_low=1, f_high=10) 
    #define array of parameters delta_m and p
    delta_m_list = np.arange(.1,2.1,.5) 
    p_list = np.arange(.1,2.1,.5)
    #get signal 
    nongausian_nonsttaionary_signal_psd = nonstationary_signal(N,PSD,fs,k_u=5,modulating_signal=('PSD', PSD_modulating),param1_list=p_list,param2_list=delta_m_list)