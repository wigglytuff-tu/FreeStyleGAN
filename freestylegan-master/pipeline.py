import os, sys

def getMainDir():
    return os.path.dirname(os.path.realpath(__file__))

def upDir(x):
   return os.path.dirname(x)

sys.path.append(upDir(getMainDir()))

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib import EasyDict

import training.networks_stylegan2 as sg2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolution or fully-connected layer.

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, stdmul=0.01, weight_var='weight'):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom multipliers.
    if use_wscale:
        init_std = 1.0 / lrmul * stdmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense_layer(x, fmaps, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# implicit latent representation network

def buildImplicitLatentNetwork(
    manifold_coord_in,                      # Input: manifold coordinates
    dynamic_latent_base_in,                 # Input: Base dynamic latents (early layers)
    static_latent_in,                       # Input: Static latents (all other layers)
    manifold_coords_size    = 3,            # Manifold coordinates dimensionality
    latent_size             = 512,          # Disentangled latent dimensionality
    style_count             = 18,           # Number of styles to output
    mod_style_count         = 4,            # Number of styles to be affected by manifold_coord_in
    layer_count             = 2,            # Number of dense layers
    fmaps                   = 512,          # Number of activations per layer.
    nonlinearity            = 'lrelu',      # Activation function
    dtype                   = 'float32',    # Data type to use for activations and outputs.
    **_kwargs):                             # Ignore unrecognized keyword args.

    manifold_coord_in.set_shape([None, manifold_coords_size])
    manifold_coord_in = tf.cast(manifold_coord_in, dtype)

    dynamic_latent_base_in.set_shape([None, mod_style_count, latent_size])
    dynamic_latent_base_in = tf.cast(dynamic_latent_base_in, dtype)

    static_latent_in.set_shape([None, style_count - mod_style_count, latent_size])
    static_latent_in = tf.cast(static_latent_in, dtype)
    static_latent_in = tf.tile(static_latent_in, (tf.shape(manifold_coord_in)[0], 1, 1))

    # mapping from manifold coord to coarse-scale styles
    feats = []
    for style in range(mod_style_count):
        x = manifold_coord_in
        for idx in range(layer_count-1):
            with tf.variable_scope('Dense%d_%d' % (style, idx+1)):
                x = sg2.apply_bias_act(dense_layer(x, fmaps=fmaps, lrmul=1.), act=nonlinearity, lrmul=1.)
        with tf.variable_scope('DenseExpansion%d' % style):
            x = sg2.apply_bias_act(dense_layer(x, fmaps=latent_size, lrmul=1.), act='linear', lrmul=1.)
        feats.append(x)
    x = tf.stack(feats, axis=1)

    # add MLP output to the base
    x += dynamic_latent_base_in

    # combine with static higher layers
    x = tf.concat([x, static_latent_in], axis=1)

    return tf.identity(x, name="latents_out")

#----------------------------------------------------------------------------

# build entire network pipeline

def build(
    gan, mlp,
    staticLatents,
    batchSize=1,
    noiseStrength=None,
    pca=None):

    nodes = EasyDict()

    nodes.staticLatents = tf.Variable(tf.constant(staticLatents), name='staticLatents')
    nodes.staticLatents.set_shape([None, staticLatents.shape[1], staticLatents.shape[2]])
    nodes.dynamicLatentsBase = tf.constant(mlp["dynamicLatentsBase"], name='dynamicLatentsBase')

    print("Building implicit latent network...")
    nodes.maniCoord = tf.placeholder(tf.float32, shape=(None, 3), name="coord")
    with tf.variable_scope('ImplicitLatent'):
        nodes.latents = buildImplicitLatentNetwork(
            manifold_coord_in=nodes.maniCoord,
            dynamic_latent_base_in=nodes.dynamicLatentsBase,
            static_latent_in=nodes.staticLatents,
            fmaps=mlp["fmaps"],
            mod_style_count=mlp["modStyleCount"],
            layer_count=mlp["layerCount"],
            nonlinearity=mlp["nonlinearity"])

    if mlp["pretrained"]:
        print("Setting latent prediction variables...")
        trainableVars = [var for var in tf.trainable_variables() if 'ImplicitLatent' in var.name]
        tflib.set_vars({var: mlp[var.name] for var in trainableVars})

    nodes.cleanLatents = nodes.latents

    if noiseStrength is not None:
        noise = noiseStrength * tf.random.normal(shape=(batchSize, 18, 512))
        nodes.latents += noise
        nodes.noiseStrength = noiseStrength

    if pca is not None:
        print("Injecting PCA controls...")
        nodes.pcaComps = tf.placeholder(tf.float32, shape=(None, pca.componentCount, 512), name="PCA_Comps")
        nodes.pcaParams = tf.placeholder(tf.float32, shape=(None, pca.componentCount, 18), name="PCA_Params")

        with tf.variable_scope("PCA"):
            pcaOffset = tf.einsum('ijk,ijl->ilk', nodes.pcaComps, nodes.pcaParams)
            nodes.latents += pcaOffset

    G_kwargs = EasyDict()
    G_kwargs.randomize_noise = False

    nodes.imageOutput = gan.components.synthesis.get_output_for(nodes.latents, **G_kwargs)

    with tf.variable_scope('OutTransform'):
        nodes.displayOutput = tflib.convert_images_to_uint8(nodes.imageOutput, nchw_to_nhwc=True)
        nodes.oglOutput = 0.5 * tf.transpose(nodes.imageOutput, [0, 2, 3, 1]) + 0.5

    print("Setting noises...")
    noiseVars = [var for var in tf.all_variables() if var.name.startswith('G_synthesis/noise')]
    tflib.set_vars({var: mlp["noises"][i] for i, var in enumerate(noiseVars)})

    return nodes