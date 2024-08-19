"""
Contains the base class for the Yule-Walker method, to estimate the 
parameters of an AR model.
"""
import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import freqz
import scipy.fft as fft

class YuleWalker:
    def __init__(self,
                 signal,
                 order
                 ):
        self.x = signal
        self.order = order
        self.ac_vec = self._autocorr(self.x, self.order)

        # Solve the Toeplitz system of equations to estimate the AR parameters.
        self._solve()
        self._freq()

    def _autocorr(self, x, lag):
        """ 
        Compute the autocorrelation of a signal x up to lag.
        """
        N = len(x)
        mean = np.mean(x)
        x = x - mean
        ac = np.correlate(x, x, mode='full')
        return ac[N-1:N+lag]
    
    def _solve(self):
        """
        Solve the Toeplitz system of equations to estimate the AR parameters.
        """
        self.R = solve_toeplitz((self.ac_vec[:self.order], self.ac_vec[:self.order]), self.ac_vec[1:self.order+1])
        self.sigma2 = self.ac_vec[0] - np.dot(self.R, self.ac_vec[1:self.order+1])

    def _freq(self, N=512):
        """
        Compute the frequency response of the AR model.

        np.r_ translates slice objects to concatenation along the first axis, 
        to quickly build up arrays.
        """
        self.w, self.h = freqz(np.sqrt(self.sigma2), np.r_[1, -self.R], worN=N)
    
    def psd(self):
        """
        Compute the power spectral density of the AR model.
        """
        return self.w, np.abs(self.h)**2
    
    def predict_next_value(self, data=None):
        """ Predict the next value using the AR coefficients """
        if not np.all(data):
            data = self.x
        prediction = np.dot(self.R, data[-len(self.R):][::-1])
        return prediction