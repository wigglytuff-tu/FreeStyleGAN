import tensorflow as tf
import numpy as np
from scipy.io import loadmat

#=======================================================

# lift input images to VGG-face feature space
# expects unnormalized images in [0, 255]
# code modified from https://github.com/ZZUTK/Tensorflow-VGG-face
def vgg_face_features(param_path, input_maps, output_layer='fc7'):

    data = loadmat(param_path)

    # read meta info
    meta = data['meta']
    normalization = meta['normalization']
    average_color = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
    image_size = np.squeeze(normalization[0][0]['imageSize'][0][0]) # 224 x 224 x 3
    input_maps = tf.image.resize(input_maps, image_size[0:2])

    input_maps -= average_color

    # read layer info
    layers = data['layers']
    current = input_maps
    for layer in layers[0]:
        name = layer[0]['name'][0][0]
        layer_type = layer[0]['type'][0][0]
        if layer_type == 'conv':
            if name[:2] == 'fc':
                padding = 'VALID'
            else:
                padding = 'SAME'
            stride = layer[0]['stride'][0][0]
            kernel, bias = layer[0]['weights'][0][0]
            bias = np.squeeze(bias).reshape(-1)
            conv = tf.nn.conv2d(current, tf.constant(kernel),
                                strides=(1, stride[0], stride[0], 1), padding=padding)
            current = tf.nn.bias_add(conv, bias)
        elif layer_type == 'relu':
            current = tf.nn.relu(current)
        elif layer_type == 'pool':
            stride = layer[0]['stride'][0][0]
            pool = layer[0]['pool'][0][0]
            current = tf.nn.max_pool2d(current, ksize=(1, pool[0], pool[1], 1),
                                     strides=(1, stride[0], stride[0], 1), padding='SAME')
        if name == output_layer:
            break
    
    output = current[::, 0, 0, ::]

    # l2 normalization, following https://www.vlfeat.org/matconvnet/pretrained/
    output = tf.math.l2_normalize(output, axis=1)

    return output
