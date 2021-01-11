pyExSi - Excitation signals as used in structural dynamics and vibration fatigue
--------------------------------------------------------------------------------
Supported excitation signals are:

- pulse (e.g. half-sine)
- random:

    - uniform random distribution
    - normal random distribution
    - pseudorandom distribution

- random, defined by power spectral density (PSD):

    - stationary Gaussian
    - stationary non-Gaussian
    - non-stationary non-Gaussian random process

- burst random
- sine sweep


Simple example
---------------

A simple example on how to generate random signals on PSD basis:

.. code-block:: python

    import pyExSi as es
    import numpy as np


    N = 2**16 # number of data points of time signal
    fs = 1024 # sampling frequency [Hz]
    t = np.arange(0,N)/fs # time vector

    # define frequency vector and one-sided flat-shaped PSD
    M = N//2 + 1 # number of data points of frequency vector
    freq = np.arange(0, M, 1) * fs / N # frequency vector
    freq_lower = 50 # PSD lower frequency limit  [Hz]
    freq_upper = 100 # PSD upper frequency limit [Hz]
    PSD = es.get_psd(freq, freq_lower, freq_upper) # one-sided flat-shaped PSD

    #get gaussian stationary signal
    gausian_signal = es.random_gaussian((N, PSD, fs)

    #get non-gaussian non-stationary signal, with kurtosis k_u=10
    #amplitude modulation, modulating signal defined by PSD
    PSD_modulating = es.get_psd(freq, freq_lower=1, freq_upper=10) 
    #define array of parameters delta_m and p
    delta_m_list = np.arange(.1,2.1,.5) 
    p_list = np.arange(.1,2.1,.5)
    #get signal 
    nongaussian_nonstationary_signal = es.nonstationary_signal(N,PSD,fs,k_u=5,modulating_signal=('PSD', PSD_modulating),param1_list=p_list,param2_list=delta_m_list)

|DOI|
|Build Status|

.. |Build Status| image:: https://travis-ci.com/ladisk/pyExSi.svg?branch=main
   :target: https://travis-ci.com/ladisk/pyExSi
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4431844.svg
   :target: https://doi.org/10.5281/zenodo.4431844