#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
def build_encoder(PARAMS):
    x = layers.Input((PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']))
    noise = layers.Input((PARAMS['latent_dim']))
    
    h = layers.Conv2D(filters = 16, kernel_size = 3, strides=2, padding = 'same')(x)
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU(0.2)(h)    
    h = layers.Conv2D(filters = 32, kernel_size = 3, strides=2, padding = 'same')(h)
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU(0.2)(h)
    
    h = layers.Flatten()(h)
    h = layers.Dense(512)(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(PARAMS['latent_dim'])(h)
    h = layers.LeakyReLU(0.2)(h)
    
    noise_h = layers.Dense(PARAMS['latent_dim'])(noise)
    noise_h = layers.LeakyReLU(0.2)(noise_h)
    
    h = h * noise_h
    
    E = K.models.Model([x, noise], h)
    # E.summary()
    
    return E
#%%
def build_generator(PARAMS):
    z = layers.Input(PARAMS['latent_dim'])
    y = layers.Input((PARAMS['class_num']))
    
    hz = layers.Dense(PARAMS['latent_dim'], input_shape = [PARAMS['latent_dim']])(z)
    hz = layers.LeakyReLU(0.2)(hz)
    hz = layers.Dense(PARAMS['latent_dim'], input_shape = [PARAMS['latent_dim']])(hz)
    hz = layers.LeakyReLU(0.2)(hz)
    
    hy = layers.Dense(PARAMS['latent_dim'])(y)
    hy = layers.LeakyReLU(0.2)(hy)
    
    h = layers.Dense(8*8*8)(layers.Concatenate()([hz, hy]))
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(8*8*32)(h)
    h = layers.LeakyReLU(0.2)(h)
    
    h = tf.reshape(h, [-1, 8, 8, 32])
    h = layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same')(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(h)
    h = layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same')(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(h)
    
    h = layers.Conv2D(filters = 3, kernel_size = 1, activation='sigmoid', padding = 'same')(h)
    
    G = K.models.Model([z, y], h)
    # G.summary()
    
    return G
#%%
def build_image_discriminator(PARAMS):
    x = layers.Input([PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']])

    h = layers.Conv2D(filters = 8, kernel_size = 3, strides = 2, padding = 'same')(x)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Flatten()(h)

    h = layers.Dense(512)(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(128)(h)
    h = layers.LeakyReLU(0.2)(h)

    h = layers.Dense(1, activation='sigmoid')(h)

    D = K.models.Model(inputs = x, outputs = h)
    # D.summary()

    return D
#%%
def build_image_classifier(PARAMS):
    x = layers.Input([PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']])

    h = layers.Conv2D(filters = 8, kernel_size = 3, strides = 2, padding = 'same')(x)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Flatten()(h)

    h = layers.Dense(512)(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(128)(h)
    h = layers.LeakyReLU(0.2)(h)

    h = layers.Dense(PARAMS['class_num'], activation='softmax')(h)

    D = K.models.Model(inputs = x, outputs = h)
    # D.summary()

    return D
#%%
def build_z_discriminator(PARAMS):
    x = layers.Input([PARAMS['latent_dim']])

    h = layers.Dense(1024)(x)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(512)(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(256)(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(128)(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(64)(h)
    h = layers.LeakyReLU(0.2)(h)

    h = layers.Dense(1, activation='sigmoid')(h)

    D = K.models.Model(inputs = x, outputs = h)
    # D.summary()

    return D
#%%