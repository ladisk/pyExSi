signal_generation.py
---------------------------------------------

Obtaining random signal, defined by power spectral density (PSD). 
Following types of random signal are  obtainable:
- stationary Gaussian random process
- stationary non-Gaussian random process
- non-stationary non-Gaussian random process.



Simple example
---------------

A simple example on how to use the code:

.. code-block:: python

    import signal_generation as sg
    import numpy as np


    N = 2**16 # number of data points of time signal
    fs = 1024 # sampling frequency [Hz]
    t = np.arange(0,N)/fs # time vector

    # define frequency vector and one-sided flat-shaped PSD
    M = int(N/2 + 1) # number of data points of frequency vector
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
    nongausian_nonsttaionary_signal_psd = sg.get_nonstationary_signal_psd(PSD,PSD_modulating,fs,N,delta_m_list,p_list, k_u=10, variance=10)