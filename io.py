import os
import numpy as np



def binary_to_array_read(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    frame_number = features.size // dimension
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))
    return features, frame_number


def array_to_binary_file(data, output_file_name):
    data = np.array(data, 'float32')
    save_file = os.path.splitext(output_file_name)[0]
    fid = open(save_file, 'wb')
    data.tofile(fid)
    fid.close()