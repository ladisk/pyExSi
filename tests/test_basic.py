import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import numpy as np
import pickle
import pyExSi as es


def test_version():
    """ check if pyExSi exposes a version attribute """
    assert hasattr(es, '__version__')
    assert isinstance(es.__version__, str)


def test_data_nonstationarity():

    with open('./tests/test_data_nonstationarity.pkl','rb') as the_file:
        test_data = pickle.load(the_file)

    results_ref = {
        'PSD': test_data['PSD'],
        'PSD modulating': test_data['PSD modulating'],
        'stationary Gaussian': test_data['stationary Gaussian'],
        'stationary nonGaussian': test_data['stationary nonGaussian'],
        'nonstationary nonGaussian PSD': test_data['nonstationary nonGaussian_PSD'],
        'nonstationary nonGaussian CSI': test_data['nonstationary nonGaussian CSI'] 
    }

    #input data
    N = test_data['N'] 
    fs = test_data['fs'] 
    f = test_data['freq']
    f_min = test_data['f_min']
    f_max = test_data['f_max']
    seed = test_data['seed']

    results = {}
    results['PSD'] = es.get_psd(f, f_min, f_max) # one-sided flat-shaped PSD

    #Random Generator 
    rg = np.random.default_rng(seed)

    #stationary Gaussian signal
    results['stationary Gaussian'] = es.random_gaussian(N, results['PSD'], fs, rg=rg)

    #stationary non-Gaussian signal
    k_u_target = 10
    rng = np.random.default_rng(seed)
    results['stationary nonGaussian'] = es.stationary_nongaussian_signal(N, results['PSD'], fs, k_u=k_u_target, rg=rg)

    #get non-gaussian non-stationary signal, with kurtosis k_u=10
    #a) amplitude modulation, modulating signal defined by PSD
    rng = np.random.default_rng(seed)
    results['PSD modulating'] = es.get_psd(f, f_low=1, f_high=k_u_target) 
    #define array of parameters delta_m and p
    delta_m_list = test_data['delta m list']
    p_list = test_data['p list']
    results['nonstationary nonGaussian PSD'] = es.nonstationary_signal(N,results['PSD'],fs,k_u=k_u_target,modulating_signal=('PSD',results['PSD modulating']),
                                                            param1_list=delta_m_list,param2_list=p_list,seed=seed)

    #b) amplitude modulation, modulating signal defined by cubis spline interpolation. Points are based on beta distribution
    #Points are separated by delta_n 
    delta_n = test_data['delta_n']
    #define array of parameters alpha and beta
    alpha_list = test_data['alpha list']
    beta_list = test_data['beta list']
    results['nonstationary nonGaussian CSI']  = es.nonstationary_signal(N,results['PSD'],fs,k_u=k_u_target,modulating_signal=('CSI',delta_n),
                                                            param1_list=alpha_list,param2_list=beta_list,seed=seed)

    for key in results.keys():
        print(key)
        np.testing.assert_almost_equal(results[key], results_ref[key], decimal=5, err_msg=f'Function: {key}')


def test_data_signals():

    with open('./tests/test_data_signals.pkl','rb') as the_file:
        test_data = pickle.load(the_file)

    results_ref = {
        'uniform random': test_data['uniform_random'],
        'normal random': test_data['normal_random'],
        'pseudo random': test_data['pseudo_random'],
        'burst random': test_data['burst_random'],
        'sweep': test_data['sweep'],
        'impact pulse_cosine': test_data['impact pulse_cosine'], 
        'impact pulse_rectangular': test_data['impact pulse_rectangular'], 
        'impact pulse_sawtooth': test_data['impact pulse_sawtooth'],
        'impact pulse_triangular': test_data['impact pulse_triangular'],
        'impact pulse_exponential': test_data['impact pulse_exponential']
    }

    #input data
    seed = test_data['seed']
    N = test_data['N'] 

    results = {}
    #Random Generator 
    rg = np.random.default_rng(seed)

    #uniform random, normal random and pseudo random
    results['uniform random'] = es.uniform_random(N=N, rg=rg)
    results['normal random'] = es.normal_random(N=N, rg=rg)
    results['pseudo random'] = es.pseudo_random(N=N, rg=rg)

    #burst random
    amplitude = test_data['burst_random amplitude'] 
    ratio = test_data['burst_random ratio']
    distribution = test_data['burst_random distribution']
    n_bursts = test_data['burst_random n_bursts']
    results['burst random'] = es.burst_random(N=N, A=amplitude, ratio=ratio, distribution=distribution, n_bursts=n_bursts, rg=rg)

    #sweep
    f_start = test_data['sweep f_start']
    f_stop = test_data['sweep f_stop']
    t = test_data['sweep t']
    results['sweep'] = es.sweep(time=t, f_start=f_start, f_stop=f_stop)

    #impact pulse 
    width = test_data['impact width']
    N = test_data['impact N']
    n_start = test_data['impact n_start']
    amplitude = test_data['impact pulse_amplitude']
    results['impact pulse_cosine']  = es.impact_pulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window='cosine')
    results['impact pulse_rectangular']  = es.impact_pulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window='boxcar')
    results['impact pulse_triangular'] = es.impact_pulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window='triang')
    results['impact pulse_exponential'] = es.impact_pulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window=('exponential',None,10))
    results['impact pulse_sawtooth'] = es.impact_pulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window='sawtooth')

    for key in results.keys():
        print(key)
        np.testing.assert_almost_equal(results[key], results_ref[key], decimal=5, err_msg=f'Function: {key}')


if __name__ == "__main__":
    test_data_signals()

if __name__ == '__mains__':
    np.testing.run_module_suite()