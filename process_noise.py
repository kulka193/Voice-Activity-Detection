import random
import os

from scipy.io import wavfile
import math_helper
noises = ['OOFFICE','DLIVING','DKITCHEN','DWASHING']
random.seed(random.randint(0,12345))

def read_noise_files(filepath):
    fs,noise_wavs = wavfile.read(filepath)
    print('file read..')
    ret_arr = math_helper.normalize(noise_wavs)
    return ret_arr

def pick_random_recording():
    real_noise = random.choice(noises)
    print('picked dataset',real_noise)
    return real_noise


def pick_random_noise_file(real_noise):
    wav_seq = [str(i).zfill(2) for i in range(1, 17)]
    noise_wavfile = str(real_noise) + '/ch' + random.choice(wav_seq) + '.wav'
    print('picked file: ',noise_wavfile)
    return noise_wavfile
