import numpy as np
import tensorflow as tf


def time_to_batch(inputs, rate):
    '''将1d信号变换为batch形式

    在1D dilated convolution. 前使用
    
    Args:
      inputs: (tensor) 
      rate: (int)
    Outputs:
      outputs: (tensor)
      pad_left: (int)
    '''
    _, width, num_channels = inputs.get_shape().as_list()

    width_pad = int(rate * np.ceil((width + rate) * 1.0 / rate))
    pad_left = width_pad - width

    perm = (1, 0, 2)
    shape = (int(width_pad / rate), -1, num_channels) # missing dim: batch_size * rate
    padded = tf.pad(inputs, [[0, 0], [pad_left, 0], [0, 0]])
    transposed = tf.transpose(padded, perm)
    reshaped = tf.reshape(transposed, shape)
    outputs = tf.transpose(reshaped, perm)
    return outputs

def batch_to_time(inputs, rate, crop_left=0):
    ''' 将运算后的信号变为1d
    
    在1D dilated convolution. 后使用
    
    Args:
      inputs: (tensor)
      crop_left: (int)
      rate: (int)
    Ouputs:
      outputs: (tensor)
    '''
    shape = tf.shape(inputs)
    batch_size = shape[0] / rate
    width = shape[1]
    
    out_width = tf.to_int32(width * rate)
    _, _, num_channels = inputs.get_shape().as_list()
    
    perm = (1, 0, 2)
    new_shape = (out_width, -1, num_channels) # missing dim: batch_size
    transposed = tf.transpose(inputs, perm)    
    reshaped = tf.reshape(transposed, new_shape)
    outputs = tf.transpose(reshaped, perm)
    cropped = tf.slice(outputs, [0, crop_left, 0], [-1, -1, -1])
    return cropped

def conv1d(inputs,
           out_channels,
           filter_width=2,
           stride=1,
           padding='VALID',
           data_format='NWC',
           gain=np.sqrt(2),
           activation=tf.nn.relu,
           bias=False,
           name='conv'):
    '''卷积函数
    
    设置好参数
    
    Args:
      inputs:
      out_channels:
      filter_width:
      stride:
      paddding:
      data_format:
      gain:
      activation:
      bias:
      name
      
    Outputs:
      outputs:
    '''
    in_channels = inputs.get_shape().as_list()[-1]

    stddev = gain / np.sqrt(filter_width**2 * in_channels)
    w_init = tf.random_normal_initializer(stddev=stddev)

    '''
    One dimensional data: 'NWC' (default) and 'NCW'
    two dimensional data: 'NHWC' (default) and 'NCHW'
    three dimensional data: 'NDHWC'
    '''
    w = tf.get_variable(name=name + '_w',
                        shape=(filter_width, in_channels, out_channels),
                        initializer=w_init)

    outputs = tf.nn.conv1d(inputs,
                           w,
                           stride=stride,
                           padding=padding,
                           data_format=data_format,
                           name=name)

    if bias:
        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable(name=name + '_b',
                            shape=(out_channels, ),
                            initializer=b_init)

        outputs = tf.add(outputs, tf.expand_dims(tf.expand_dims(b, 0), 0), name=name + '_add')

    if activation:
        outputs = activation(outputs)

    return outputs

def dilated_conv1d(inputs,
                   out_channels,
                   filter_width=2,
                   rate=1,
                   padding='VALID',
                   name='dilated',
                   gain=np.sqrt(2),
                   activation=tf.nn.relu):
    '''

    因果卷积

    Args:
      inputs: (tensor)
      output_channels:
      filter_width:
      rate:
      padding:
      name:
      gain:
      activation:

    Outputs:
      outputs: (tensor)
    '''
    assert name
    with tf.variable_scope(name):
        _, width, _ = inputs.get_shape().as_list()
        inputs_ = time_to_batch(inputs, rate=rate)
        outputs_ = conv1d(inputs_,
                          out_channels=out_channels,
                          filter_width=filter_width,
                          padding=padding,
                          gain=gain,
                          activation=activation,
                          bias=True)
        _, conv_out_width, _ = outputs_.get_shape().as_list()
        new_width = conv_out_width * rate
        diff = new_width - width
        outputs = batch_to_time(outputs_, rate=rate, crop_left=diff)

        # Add additional shape information.
        tensor_shape = [tf.Dimension(None),
                        tf.Dimension(width),
                        tf.Dimension(out_channels)]
        outputs.set_shape(tf.TensorShape(tensor_shape))

    return outputs

def dense_layers(inputs,
                out_channels,
                filter_width=1,
                padding='SAME',
                bias=True,
                name='dense',
                gain=np.sqrt(2),
                activation=tf.nn.relu):
    assert name
    with tf.variable_scope(name):
        outputs = conv1d(inputs,
                         out_channels,
                         filter_width=filter_width,
                         padding=padding,
                         gain=gain,
                         activation=activation,
                         bias=bias)
        return outputs

def skip_layers(inputs,
                out_channels,
                filter_width=1,
                padding='SAME',
                bias=True,
                name='skip',
                gain=np.sqrt(2),
                activation=tf.nn.relu):
    assert name
    with tf.variable_scope(name):
        outputs = conv1d(inputs,
                         out_channels,
                         filter_width=filter_width,
                         padding=padding,
                         gain=gain,
                         activation=activation,
                         bias=bias)
        return outputs


def _pretreatment(inputs, name='pretreatment', activation=tf.nn.tanh):
    assert name
    with tf.variable_scope(name, reuse=True):
        w = tf.get_variable('conv_w')
        w_ = w[0, :, :]
        b = tf.get_variable('conv_b')

        output = tf.matmul(inputs, w_) + b

        if activation:
            output = activation(output)
    return output


def _causal_linear(inputs, state, name=None, activation=tf.nn.tanh):
    assert name
    with tf.variable_scope(name, reuse=True):
        w = tf.get_variable('dilated/conv_w')
        b = tf.get_variable('dilated/conv_b')

        w_r = w[0, :, :]
        w_e = w[1, :, :]

        output = tf.matmul(inputs, w_e) + tf.matmul(state, w_r) + b

        if activation:
            output = activation(output)
    return output

def _skip(inputs, name=None, activation=tf.nn.relu):
    assert name
    with tf.variable_scope(name, reuse=True):
        w = tf.get_variable('skip/conv_w')[0, :, :]
        b = tf.get_variable('skip/conv_b')

        output = tf.matmul(inputs, w) + b

        if activation:
            output = activation(output)
    return output

def _dense(inputs, name=None, activation=tf.nn.relu):
    assert name
    with tf.variable_scope(name, reuse=True):
        w = tf.get_variable('dense/conv_w')[0, :, :]
        b = tf.get_variable('dense/conv_b')

        output = tf.matmul(inputs, w) + b

        if activation:
            output = activation(output)
    return output

def _output_linear(h, name='processing'):
    with tf.variable_scope(name, reuse=True):
        w_1 = tf.get_variable('out_conv1_w')[0, :, :]
        b_1 = tf.get_variable('out_conv1_b')
        w_2 = tf.get_variable('out_conv2_w')[0, :, :]
        b_2 = tf.get_variable('out_conv2_b')

        h = tf.matmul(h, w_1) + tf.expand_dims(b_1, 0)
        h = tf.nn.relu(h)

        output = tf.matmul(h, w_2) + tf.expand_dims(b_2, 0)
    return output
