from wavenet.utils import make_batch
from wavenet.models import Model, Generator
from IPython.display import Audio
from wavenet.utils import mu_law_decode
import time
import numpy as np
import librosa
import os

data, label = make_batch('assets/data/')
num_time_samples = int(16000 * 4)

num_channels = 1
gpu_fraction = 0.75

model = Model(num_time_samples=num_time_samples,
              num_channels=num_channels,
              gpu_fraction=gpu_fraction)

# Audio(inputs.reshape(inputs.shape[1]), rate=16000)

tic = time.time()
model.train(num_time_samples,
            data,
            label,
            restoredir='./logdir/2019-02-17T23-16-00',
            savedir='./logdir',
            terminal=False,
            temperature_flg=False)
toc = time.time()

print('Training took {} seconds.'.format(toc-tic))

# TODO 生成方法待检查
generator = Generator(model)

# Get first sample of input

num = np.random.randint(0, len(data))
start = np.random.randint(0, int(data[num].shape[1]))
input_ = data[num][:, start, :]

tic = time.time()
predictions = generator.run(input_, 160000)
toc = time.time()
print('Generating took {} seconds.'.format(toc-tic))

predictions = mu_law_decode(predictions)
Audio(predictions, rate=16000)

Time = time.strftime('%Y-%m-%dT%H-%M-%S',time.localtime(time.time()))
generator_dir = './generator/' + str(Time)
os.makedirs(generator_dir)
generator_dir = generator_dir + '/test.wav'
y = np.array(predictions[0, :])
y = y.astype(np.float32)
librosa.output.write_wav(generator_dir, y, 16000)