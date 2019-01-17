import numpy as np

from scipy.io import wavfile


def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path):
    data = wavfile.read(path)[1][:, 0]

    data_ = normalize(data)
    data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))

    bins = np.linspace(-1, 1, 256)
    # 量化输入
    inputs = np.digitize(data_f[0:-1], bins, right=False) - 1
    inputs = bins[inputs][None, :, None]

    # targets数字化
    targets = (np.digitize(data_f[1::], bins, right=False) - 1)[None, :]
    return inputs, targets
