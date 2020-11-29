import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import numpy as np
import matplotlib.pyplot as plt
import pyExSi as es

#impact pulse 
width = 300 
N = 2*width
n_start = 90
amplitude = 3
pulse_cosine = es.impact_pulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window='cosine')
pulse_rectangular = es.impact_pulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window='boxcar')
pulse_triangular = es.impact_pulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window='triang')
pulse_exponential = es.impact_pulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window=('exponential',None,10))
pulse_sawtooth = es.impact_pulse(N=N, n_start=n_start, width=width, amplitude=amplitude, window='sawtooth')
plt.plot(pulse_cosine, '-', label='cosine')
plt.plot(pulse_rectangular, '-', label='rectangular')
plt.plot(pulse_triangular, '-', label = 'triangular')
plt.plot(pulse_exponential, '-', label = 'exponential')
plt.plot(pulse_sawtooth, '-', label = 'sawtooth')
plt.xlabel('Sample [n]')
plt.ylabel('Impact pulse [Unit]')
plt.legend(loc='upper right')
plt.show()


#Random Generator 
seed = 1234
rg = np.random.default_rng(seed) #or rng = np.random.default_rng(seed) 

#uniform random, normal random and pseudo random
N = 1000
uniform_random = es.uniform_random(N=N,rg=rg)
normal_random = es.normal_random(N=N,rg=rg)
pseudo_random = es.pseudo_random(N=N,rg=rg)
plt.plot(uniform_random, label='uniform random')
plt.plot(normal_random, label='normal random')
plt.plot(pseudo_random, label='pseudo random')
plt.xlabel('Sample [n]')
plt.ylabel('Random signal [Unit]')
plt.legend()
plt.show()
print(uniform_random[:5])


#burst random
N = 1000
amplitude = 5
burst_random = es.burst_random(N, A=amplitude, ratio=0.1, distribution='normal', n_bursts=3, rg=rg)
plt.plot(burst_random)
plt.xlabel('Sample [n]')
plt.ylabel('Burst [Unit]')
plt.show()


#sweep
t = np.linspace(0,10,1000)
sweep = es.sweep(time=t, f_start=0, f_stop=5)
plt.plot(t, sweep)
plt.xlabel('Time [s]')
plt.ylabel('Sweep [Unit]')
plt.show()