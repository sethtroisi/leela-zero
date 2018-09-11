import tensorflow as tf
import numpy as np
import sys

if len(sys.argv) < 2:
    print('Model filename without extension needed as an argument.')
    exit()

sess = tf.Session()
saver = tf.train.import_meta_graph(sys.argv[1]+'.meta')
saver.restore(sess, sys.argv[1])

def matches(name, parts):
    return all(part in name for part in parts)

if 0:
    # Exports graph to tensorboard
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        writer.close()

trainable_names = []
for v in tf.trainable_variables():
    trainable_names.append(v.name)

var_names = tf.train.load_checkpoint(sys.argv[1]).get_variable_to_dtype_map()
print (sorted(var_names.keys()))
print (len(var_names))

print (sorted(trainable_names))
print (len(trainable_names))


has_conv_bias = any(matches(name, ('conv2d', 'bias')) for name in trainable_names)
if not has_conv_bias:
    print('Did not find conv bias in this model, using all zeros')
empty_conv_bias = tf.constant([], name='placeholder_for_conv_bias')



weights = []
for v in tf.global_variables():
    if v.name in trainable_names:
        weights.append(v)
        if matches(v.name, ('conv2d', 'kernel')) and not has_conv_bias:
            # Newer models of Minigo have removed conv2d kernel.
            weights.append(empty_conv_bias)
    elif 'batch_normalization' in v.name:
        # Moving mean and variance are not trainable, but are needed for the model
        if 'moving_mean' in v.name or 'moving_variance' in v.name:
            weights.append(v)

if 0:
    for w in weights:
        print(w.name, w)

def merge_gammas(weights):
    out_weights = []
    skip = 0
    for e, w in enumerate(weights):
        if skip > 0:
            skip -= 1
            continue

        if matches(w.name, ('conv2d', 'kernel')) and not has_conv_bias:
            assert weights[e+1] == empty_conv_bias
            weights[e+1] = tf.zeros(w.shape[-1:])

        if matches(w.name, ('conv2d', 'kernel')) and 'gamma' in weights[e+2].name:
            kernel = w
            bias = weights[e+1]
            gamma = weights[e+2]
            beta = weights[e+3]
            mean = weights[e+4]
            var = weights[e+5]

            new_kernel = kernel * tf.reshape(gamma, (1, 1, 1, -1))
            new_bias = gamma * bias + beta * tf.sqrt(var + tf.constant(1e-5))
            new_mean = mean * gamma

            out_weights.append(new_kernel)
            out_weights.append(new_bias)
            out_weights.append(new_mean)
            out_weights.append(var)

            skip = 5
        elif matches(w.name, ('dense', 'kernel')):
            # Minigo uses channels last order while LZ uses channels first,
            # Do some surgery for the dense layers to make the output match.
            planes = w.shape[0].value//361
            if planes > 0:
                w1 = tf.reshape(w, [19, 19, planes, -1])
                w2 = tf.transpose(w1, [2, 0, 1, 3])
                new_kernel = tf.reshape(w2, [361*planes, -1])
                out_weights.append(new_kernel)
            else:
                out_weights.append(w)
        else:
            #print (w)
            out_weights.append(w)

    return out_weights

def save_leelaz_weights(filename, weights):
    with open(filename, 'w') as file:
        # Version tag
        # Minigo outputs winrate from blacks point of view (same as ELF)
        file.write('2')
        for e, w in enumerate(weights):
            # Newline unless last line (single bias)
            file.write('\n')
            work_weights = None
            if w.shape.ndims == 4:
                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                work_weights = tf.transpose(w, [3, 2, 0, 1])
            elif w.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                work_weights = tf.transpose(w, [1, 0])
            else:
                # Biases, batchnorm etc
                work_weights = w
            nparray = work_weights.eval(session=sess)
            if e == 0:
                # Fix input planes

                # Add zero weights for white to play input plane
                nparray = np.pad(nparray, ((0, 0), (0, 1), (0, 0), (0, 0)), 'constant', constant_values=0)

                # Permutate weights
                p = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17]

                nparray = nparray[:, p, :, :]
            print (e, nparray.shape)
            wt_str = [str(wt) for wt in np.ravel(nparray)]
            file.write(' '.join(wt_str))

save_leelaz_weights(sys.argv[1]+'_converted.txt', merge_gammas(weights))
