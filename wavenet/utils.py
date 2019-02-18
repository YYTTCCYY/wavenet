import numpy as np
import os
import librosa
from scipy.io import wavfile


def normalize(data, threshold=0):
    threshold = data > threshold
    data = data * threshold.astype(np.int32)

    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path):
    all_file_name = os.listdir(path)
    date_dict = dict()
    label_dict = dict()
    for i, file_name in enumerate(all_file_name):
        file_name = path + file_name
        data = librosa.load(file_name, sr=16000)
        data = np.array(data[0])

        data_ = normalize(data)
        data_f = mu_law_code(data_)

        bins = np.linspace(-1, 1, 256)
        # 量化输入
        data = np.digitize(data_f[0:-1], bins, right=False) - 1
        data = bins[data][None, :, None]

        # label数字化
        label = (np.digitize(data_f[1::], bins, right=False) - 1)[None, :]
        date_dict[i] = data
        label_dict[i] = label
    return date_dict, label_dict

def mu_law_code(data):
    return np.sign(data) * (np.log(1 + 255 * np.abs(data)) / np.log(1 + 255))

def mu_law_decode(data):
    return np.sign(data) * ((255 + 1) ** np.abs(data) - 1) / 255