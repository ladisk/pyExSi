"""
This is only meant to demonstrate the agreement of the `sweep` implementation 
with `scipy.signal.chirp`. Can be used for unit testing of our sine sweep 
implementation down the road, but it also shows that the current `sine_sweep` 
function could be reworked to only call `scipy.signal.sweep` with minimal effort.
"""
import sys, os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

from scipy.signal import chirp
import pyExSi as es
import numpy as np


def test_chirp_vs_scipy(plot=False):
    if plot:
        import matplotlib.pyplot as plt

    ts = np.linspace(0, 1, 1000)
    f0 = 10
    f1 = 100
    t1 = ts[-1]
    method = 'linear'
    phi = 0
    phi_scipy = (phi - np.pi / 2) / np.pi * 180

    results = []
    for method in ['linear', 'logarithmic']:

        s_scipy = chirp(ts, f0, t1, f1, phi=phi_scipy, method=method)
        s_own = es.sine_sweep(ts, phi, f_start=f0, f_stop=f1, mode=method)
        results.append((s_scipy, s_own))

        if plot:
            fig, ax = plt.subplots(2, 1)
            ax[0].set_title(f'time signal, mode={method}')
            ax[0].plot(ts, s_scipy, label='scipy')
            ax[0].plot(ts, s_own, '--', label='own')

            S_scipy = np.fft.rfft(s_scipy) / len(ts) * 2
            S_own = np.fft.rfft(s_own) / len(ts) * 2
            freq_s = np.fft.rfftfreq(len(ts), ts[1] - ts[0])

            ax[1].set_title(f'amplitude spectrum, mode={method}')
            ax[1].plot(freq_s, np.abs(S_scipy), label='scipy')
            ax[1].plot(freq_s, np.abs(S_own), '--', label='own')
            ax[1].axvline(x=f0, color='k')
            ax[1].axvline(x=f1, color='k')
            ax[1].set_xlim(f0 - 10, f1 + 5)
            ax[1].legend()
            plt.tight_layout()

    if plot:
        plt.show()

    for r in results:
        np.testing.assert_allclose(*r, atol=1e-10)


if __name__ == "__main__":
    test_chirp_vs_scipy(plot=True)

if __name__ == '__mains__':
    np.testing.run_module_suite()
