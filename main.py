import os
import sys
import numpy as np

import soundfile as sf
import matplotlib.pyplot as plt
import process_noise
import math_helper
import random
from feature_extract import MFCC
import io
import NN_main



print('currently in :',os.listdir())


all_files = os.listdir('LibriSpeech/dev-clean/')

BUFFER_SIZE = 700000


def read_flacs(flacfile):
    data , fs = sf.read(flacfile)
    return (data,fs)



def get_all_audios():
    all_flacs = []
    for f in os.listdir():
        print('parent: ',f)
        for f1 in os.listdir(f):
            print("child: ",f1)
            for flacs in os.listdir(os.path.join(f,f1)):
                #print("flacs: ",flacs)
                all_flacs.append(os.path.join(f,f1,flacs))
    return all_flacs


def visualize_sample_audio(data):
    plt.xlabel('time')
    plt.ylabel('amplitude/gain')
    plt.title('time waveform for a sample flac audio file')

    plt.plot(data)
    plt.show()





def divide_frames_each_file(data, fs=16000):
    window_dur = 32e-3
    num_samples = data.shape[0]
    audio_dur = num_samples / fs
    num_of_frames = int(np.floor(audio_dur / window_dur))
    frame_length = int(window_dur * fs)

    samples_left = num_samples - (num_of_frames * frame_length)
    leading_zeros = np.zeros((abs(samples_left - frame_length),))

    padded_data = np.concatenate((data, leading_zeros), axis=0)
    frames_divided = np.reshape(padded_data, [-1, frame_length])

    return frames_divided


def noise_wrapper():
    noise_wavfile = process_noise.pick_random_noise_file(real_noise=process_noise.pick_random_recording())
    noisy = process_noise.read_noise_files(os.path.join(os.getcwd(), noise_wavfile))
    normalized_noisy = math_helper.normalize(data=noisy)
    return divide_frames_each_file(data=normalized_noisy)





def prepare_input_files(all_flacs,NFFT,fs,n_dct):
    summ_frames = 0
    v_labels = np.empty((BUFFER_SIZE,))
    alpha = 1
    if not os.path.exists(path=os.path.join(os.getcwd(),'mfcc')):
        os.mkdir(os.path.join(os.getcwd(),'mfcc'))
    mfcc = MFCC(NFFT=NFFT,fs=fs,n_dct = n_dct)
    for i,flac in enumerate(all_flacs):
        print(os.path.splitext(flac))
        if os.path.splitext(flac)[1] == ".flac":
            st_index = random.randint(0,8000)
            flacpath = os.path.join('LibriSpeech/dev-clean/',flac)
            data,fs = read_flacs(flacpath)
            assert fs == 16000
            framed_data = divide_frames_each_file(data,fs)
            #print(framed_data)
            framed_noise = noise_wrapper()[st_index:st_index+framed_data.shape[0]]
            X_l =framed_data+alpha*framed_noise

            mfcc = mfcc.get_mfcc(X_l)
            io.array_to_binary_file(mfcc,os.path.join('mfcc',flac.split('/')[-1]))
            v_labels[summ_frames:summ_frames+framed_data.shape[0]] = \
                math_helper.norm_squared(framed_data)/\
                (math_helper.norm_squared(framed_data)+math_helper.norm_squared(framed_noise))
            summ_frames += framed_data.shape[0]
            print("picked and processed", summ_frames ,"frames so far: " )
    return summ_frames, v_labels


def training_wrapper_for_rnn(batch_size,num_epochs,input_file_list,out_vector,alpha,timesteps,n_dct = 60):
    if not os.path.exists(os.path.join(os.getcwd(),'rnn_model')):
        os.mkdir('rnn_model')
    #train_data_gens = [ data_generator(input_file_list, batch_size, out_vector, inp_dim = 60)
    #                   for epoch in range( num_epochs ) ]
    X_data = NN_main.get_all_frames(input_file_list,inp_dim=n_dct,total_frames = out_vector.shape[0])
    validation_error = NN_main.train_rnn(X_data,out_vector, alpha, num_epochs, timesteps,
                                            ckpt_dir = os.path.join(os.getcwd(),'rnn_model'),batch_size=batch_size)
    NN_main.plot_validation(validation_error)

def training_wrapper(batch_size,num_epochs,input_file_list,out_vector,alpha):
    if not os.path.exists(os.path.join(os.getcwd(),'model')):
        os.mkdir('model')
    #train_data_gens = [ data_generator(input_file_list, batch_size, out_vector, inp_dim = 60)
    #                   for epoch in range( num_epochs ) ]
    X_data = NN_main.get_all_frames(input_file_list,inp_dim=60,total_frames = out_vector.shape[0])
    validation_error = NN_main.train_neural_network(X_data,out_vector, alpha, num_epochs,
                                            ckpt_dir = os.path.join(os.getcwd(),'model'),
                                            batch_size=batch_size,shuffle_data=True)
    NN_main.plot_validation(validation_error)



def main():
    all_flacs = get_all_audios()
    visualize_sample_audio(read_flacs('/home/bhargav/Documents/VAD/LibriSpeech/dev-clean/' + all_flacs[0])[0])
    total_frames, v_labels = prepare_input_files(all_flacs,NFFT=1024,fs=16000,n_dct=60)
    v_labels = v_labels[:total_frames]
    training_wrapper(batch_size=1024, num_epochs=20, input_file_list=os.listdir('mfcc'),
                     out_vector=v_labels.reshape([-1, 1]), alpha=0.005)
    


if __name__=="__main__":
    main()









