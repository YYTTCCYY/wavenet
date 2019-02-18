import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import os
from wavenet.utils import (mu_law_decode)
from wavenet.layers import (_pretreatment, _causal_linear,
                            _output_linear, conv1d, _skip,
                            _dense, dilated_conv1d,
                            dense_layers, skip_layers)


class Model(object):
    def __init__(self,
                 num_time_samples,
                 num_channels=1,
                 num_classes=256,
                 num_blocks=2,
                 num_layers=14,
                 num_hidden=128,
                 gpu_fraction=1.0):
        
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gpu_fraction = gpu_fraction
        
        inputs = tf.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels))
        targets = tf.placeholder(tf.int32, shape=(None, num_time_samples))

        with tf.variable_scope('pretreatment'):
            h = conv1d(
                inputs, num_hidden, filter_width=1, activation=tf.nn.tanh, bias=True)
        hs = []
        skips = []
        cut_outwidth = num_blocks * ((2 ** (num_layers - 1)) - 1)
        '''主体模型'''
        with tf.variable_scope('main_lays'):
            for b in range(num_blocks):
                for i in range(num_layers):
                    rate = 2**i
                    name = 'b{}-l{}'.format(b, i)
                    with tf.variable_scope(name):
                        '''dilated_layers'''
                        h_ = dilated_conv1d(h,
                                            num_hidden,
                                            rate=rate,
                                            name='dilated',
                                            activation=tf.nn.tanh)
                        '''dense_layers'''
                        h_dense = dense_layers(h_,
                                               num_hidden,
                                               filter_width=1,
                                               padding="SAME",
                                               bias=True,
                                               name='dense')
                        h = h_dense + h
                        '''skip_layers'''
                        skip = tf.slice(h_, [0, cut_outwidth, 0], [-1, -1, -1], name='skip/slice')
                        skip_contribution = skip_layers(skip,
                                                   num_hidden,
                                                   filter_width=1,
                                                   padding="SAME",
                                                   bias=True,
                                                   name='skip')

                    hs.append(h)
                    skips.append(skip_contribution)
        '''后处理layers'''
        with tf.variable_scope('processing'):
            total = sum(skips)
            transformed1 = tf.nn.relu(total)
            conv1 = conv1d(transformed1,
                           num_hidden,
                           filter_width=1,
                           padding="SAME",
                           bias=True,
                           name='out_conv1')
            # conv2 = conv1d(conv1,
            #               num_hidden,
            #               filter_width=1,
            #               padding="SAME",
            #               activation=tf.nn.softmax, bias=True)
            outputs = conv1d(conv1,
                             num_classes,
                             filter_width=1,
                             gain=1.0,
                             activation=None,
                             bias=True,
                             name='out_conv2')

        '''cost_layer'''
        # outputs_slice = tf.slice(outputs, [0, cut_outwidth, 0], [-1, -1, -1])
        targets_slice = tf.slice(targets, [0, cut_outwidth], [-1, -1])
        costs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets_slice, logits=outputs)

        # costs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=targets, logits=outputs)
        cost = tf.reduce_mean(costs)
        tf.summary.scalar('cost', cost)

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.initialize_all_variables())

        self.inputs = inputs
        self.targets = targets
        self.outputs = outputs
        self.hs = hs
        self.costs = costs
        self.cost = cost
        self.train_step = train_step
        self.sess = sess
        self.summaries = tf.summary.merge_all()
        self.cut_outwidth = cut_outwidth

    def _train(self, inputs, targets):
        feed_dict = {self.inputs: inputs, self.targets: targets}
        summary, cost, output, _ = self.sess.run(
            [self.summaries, self.cost, self.outputs, self.train_step],
            feed_dict=feed_dict)
        targets = targets[:, self.cut_outwidth: ]
        max_output = np.argmax(output, axis=2)
        accuracy = max_output == targets
        accuracy = accuracy[0, :]
        accuracy = sum(accuracy)/len(accuracy)
        return summary, cost, accuracy

    # def create(self, num_time_samples, data, label):
    #     num = np.random.randint(0, len(data))
    #     start = np.random.randint(0, int(data[num].shape[1] - num_time_samples))
    #     return data[num][:, start: start + num_time_samples, :], label[num][:, start: start + num_time_samples]

    def train(self, num_time_samples,
              data,
              label,
              restoredir=None,
              savedir=None,
              terminal = False,
              temperature_flg=False):
        saver = tf.train.Saver()
        if restoredir is not None:
            ckpt = tf.train.get_checkpoint_state(restoredir)
            if ckpt is not None:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                model_num = ckpt.model_checkpoint_path
                i = int(model_num.split('\\')[-1].split('-')[-1])
                print('restore model')
            else:
                print('File not found and will going to create a file')
                i = 0
        else:
            print('File not found and will going to create a file')
            i = 0

        Time = time.strftime('%Y-%m-%dT%H-%M-%S',time.localtime(time.time()))
        savedir = savedir + '/' + str(Time)
        os.mkdir(savedir)

        writer = tf.summary.FileWriter(savedir)
        writer.add_graph(tf.get_default_graph())

        losses = []
        while not terminal:
            i += 1
            num = np.random.randint(0, len(data))
            start = np.random.randint(0, int(data[num].shape[1] - num_time_samples))
            # inputs, targets = self.create(num_time_samples, data, label)
            summary, cost, accuracy = self._train(data[num][:, start: start + num_time_samples, :]\
                                        , label[num][:, start: start + num_time_samples])
            if (temperature_flg):
                temperature = os.popen(
                    'C:\\"Program Files"\\"NVIDIA Corporation"\\NVSMI\\nvidia-smi.exe').read()
                temperature = temperature.split('N/A')
                temperature = int(temperature[2].split('C')[0].split(' ')[-1])
                while(temperature > 72):
                    temperature = os.popen(
                        'C:\\"Program Files"\\"NVIDIA Corporation"\\NVSMI\\nvidia-smi.exe').read()
                    temperature = temperature.split('N/A')
                    temperature = int(temperature[2].split('C')[0].split(' ')[-1])
                    time.sleep(1.5)

            print('step = ' + str(i) + '\n' + 'cost = ' + str(cost), 'accuracy = ' + str(accuracy))
            writer.add_summary(summary, int(i))
            if cost < 1e-1:
                print('step = ' + str(i) + '\n' + 'cost = ' + str(cost), 'accuracy = ' + str(accuracy))
                print('end and save\n')
                saver.save(self.sess, savedir + '/MyModel-' + str(i))
                terminal = True
            losses.append(cost)
            if i % 50 == 0:
                print('step = ' + str(i) + '\n' + 'cost = ' + str(cost), 'accuracy = ' + str(accuracy))
                print('save\n')
                saver.save(self.sess, savedir + '/MyModel-' + str(i), write_meta_graph=False)
                # plt.plot(losses)
                # plt.show()


class Generator(object):
    def __init__(self, model, batch_size=1, input_size=1):
        self.model = model
        self.bins = np.linspace(-1, 1, self.model.num_classes)

        inputs = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')

        print('Make Generator.')

        count = 0
        h = inputs
        h = _pretreatment(h)

        init_ops = []
        push_ops = []
        skips = []
        for b in range(self.model.num_blocks):
            for i in range(self.model.num_layers):
                rate = 2**i
                name = 'main_lays/b{}-l{}'.format(b, i)
                # if count == 0:
                #     state_size = 1
                # else:
                #     state_size = self.model.num_hidden

                state_size = self.model.num_hidden

                q = tf.FIFOQueue(rate,
                                 dtypes=tf.float32,
                                 shapes=(batch_size, state_size))
                init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

                state_ = q.dequeue()
                push = q.enqueue([h])
                init_ops.append(init)
                push_ops.append(push)

                h_ = _causal_linear(h, state_, name=name, activation=tf.nn.tanh)
                skips.append(_skip(h_, name=name))

                h = h + _dense(h_, name=name)

                count += 1

        h = sum(skips)
        outputs = _output_linear(h)

        out_ops = [tf.nn.softmax(outputs)]
        out_ops.extend(push_ops)

        self.inputs = inputs
        self.init_ops = init_ops
        self.out_ops = out_ops
        
        # Initialize queues.
        self.model.sess.run(self.init_ops)

    def run(self, input, num_samples):
        predictions = []
        for step in range(num_samples):

            feed_dict = {self.inputs: input}
            output = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0] # ignore push ops
            value = np.argmax(output[0, :])

            input = np.array(self.bins[value])[None, None]
            predictions.append(input)

            if step % 1000 == 0:
                predictions_ = np.concatenate(predictions, axis=1)
                plot_wav = mu_law_decode(predictions_)
                plt.plot(plot_wav[0, :], label='pred')
                plt.legend()
                plt.xlabel('samples from start')
                plt.ylabel('signal')
                plt.show()

        predictions_ = np.concatenate(predictions, axis=1)
        return predictions_
