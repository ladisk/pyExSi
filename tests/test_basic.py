import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import numpy as np
import matplotlib.pyplot as plt
import pickle
import signal_generation as sg


def test_version():
    """ check signal_generation exposes a version attribute """
    assert hasattr(sg, '__version__')
    assert isinstance(sg.__version__, str)


def test_data():

    with open('./test/test_data.pkl','rb') as the_file:
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

    results = {}
    results['PSD'] = sg.get_psd(f, f_min, f_max) # one-sided flat-shaped PSD

    #Random Generator seed
    seed = 0
    #stationary Gaussian signal
    rng = np.random.default_rng(seed)
    results['stationary Gaussian'] = sg.random_gaussian(N, results['PSD'], fs, rng=rng)

    #stationary non-Gaussian signal
    k_u_target = 10
    rng = np.random.default_rng(seed)
    results['stationary nonGaussian'] = sg.stationary_nongaussian_signal(N, results['PSD'], fs, k_u=k_u_target, rng=rng)

    #get non-gaussian non-stationary signal, with kurtosis k_u=10
    #a) amplitude modulation, modulating signal defined by PSD
    rng = np.random.default_rng(seed)
    results['PSD modulating'] = sg.get_psd(f, f_low=1, f_high=k_u_target) 
    #define array of parameters delta_m and p
    delta_m_list = test_data['delta m list']
    p_list = test_data['p list']
    results['nonstationary nonGaussian PSD'] = sg.nonstationary_signal(N,results['PSD'],fs,k_u=k_u_target,modulating_signal=('PSD',results['PSD modulating']),
                                                            param1_list=delta_m_list,param2_list=p_list,seed=seed)

    #b) amplitude modulation, modulating signal defined by cubis spline interpolation. Points are based on beta distribution
    #Points are separated by delta_n 
    delta_n = test_data['delta_n']
    #define array of parameters alpha and beta
    alpha_list = test_data['alpha list']
    beta_list = test_data['beta list']
    results['nonstationary nonGaussian CSI']  = sg.nonstationary_signal(N,results['PSD'],fs,k_u=k_u_target,modulating_signal=('CSI',delta_n),
                                                            param1_list=alpha_list,param2_list=beta_list,seed=seed)


    for key in results.keys():
        print(key)
        np.testing.assert_almost_equal(results[key], results_ref[key], decimal=5, err_msg=f'Function: {key}')


if __name__ == "__main__":
    test_data()
    #test_version()

