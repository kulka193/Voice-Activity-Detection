
from scipy.fftpack import idct, dct, fft2, ifft2, fft,ifft
import numpy as np
import os
import math
import math_helper

class MFCC:
    def __init__(self,NFFT,fs,n_dct,f_bank_length=60):
        self.NFFT = NFFT
        self.fs = fs
        self.n_dct = n_dct
        self.f_bank_length = f_bank_length
        mel_max = math_helper.hz2mel(hz=self.fs)
        mel_scale = np.linspace(0, mel_max, self.f_bank_length)
        hz_scale = math_helper.mel2hz(mel_scale)

        self.bins = np.floor((self.NFFT) * hz_scale / self.fs)

    def compute_fft(self,time_data):
        freq_samples = fft(time_data, self.NFFT, axis=1)
        pow_samples = np.absolute(freq_samples) ** 2
        pow_samples = pow_samples / pow_samples.max()
        pow_db = np.log10(pow_samples)
        pow_db[pow_db < -5] = -5
        return pow_db

    def _prepare_mel_filter(self):
        H_filt_resp = np.zeros((self.f_bank_length, self.NFFT))
        for k in range(0, self.f_bank_length - 2):
            fm_plus = int(self.bins[k + 2])
            fm = int(self.bins[k + 1])
            fm_minus = int(self.bins[k])

            for m in range(fm_minus, fm):
                H_filt_resp[k, m] = (m - fm_minus) / (fm - fm_minus)
            for m in range(fm, fm_plus):
                H_filt_resp[k, m] = (fm_plus - m) / (fm_plus - fm)
        H_filt_resp = np.where(H_filt_resp == 0, np.finfo(float).eps, H_filt_resp)
        return H_filt_resp

    def get_mfcc(self,data_block):
        H_filt_resp = self._prepare_mel_filter()
        pow_db = self.compute_fft(data_block)
        num_of_frames = pow_db.shape[0]
        Y_bank_out = np.dot(pow_db, H_filt_resp.T)
        y_coeff = np.empty((num_of_frames, self.n_dct))
        for row in range(Y_bank_out.shape[0]):
            y_coeff[row, :] = dct(Y_bank_out[row, :], type=2, n=self.n_dct, norm='ortho')
        ncoeff = y_coeff.shape[1]
        n = np.arange(0, ncoeff)
        cep_lifter = 0.001 * ncoeff
        lift = 1 + (ncoeff) * np.sin(np.pi * n / cep_lifter)
        mfcc = y_coeff * lift
        mfcc -= (np.mean(y_coeff, axis=0))
        return mfcc


