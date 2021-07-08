#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
#%%
def build_encoder(PARAMS):
    x = layers.Input((PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']))
    
    # h = layers.Conv2D(filters = 8, kernel_size = 3, strides = 2, padding = 'same')(x)
    # h = layers.LeakyReLU(0.2)(h)
    
    h = layers.Flatten()(x)
    h = layers.Dense(1024)(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(512)(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(256)(h)
    h = layers.LeakyReLU(0.2)(h)
    
    mean = layers.Dense(PARAMS['latent_dim'])(h)
    logvar = layers.Dense(PARAMS['latent_dim'])(h)
    
    E = K.models.Model(x, [mean, logvar])
    # E.summary()
    
    return E
#%%
def build_generator(PARAMS):
    z = layers.Input(PARAMS['latent_dim'])
    y = layers.Input(PARAMS['class_num'])
    
    h = layers.Dense(7*7*256, use_bias=False)(layers.Concatenate()([z, y]))
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Reshape((7, 7, 256))(h)
    
    hy = tf.tile(y[:, tf.newaxis, tf.newaxis, :], [1, h.shape[1], h.shape[1], 1])
    h = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(layers.Concatenate()([h, hy]))
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU(0.2)(h)
    
    hy = tf.tile(y[:, tf.newaxis, tf.newaxis, :], [1, h.shape[1], h.shape[1], 1])
    h = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(layers.Concatenate()([h, hy]))
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU(0.2)(h)
    
    hy = tf.tile(y[:, tf.newaxis, tf.newaxis, :], [1, h.shape[1], h.shape[1], 1])
    h = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid')(layers.Concatenate()([h, hy]))
    
    G = K.models.Model([z, y], h)
    # G.summary()
    
    return G
#%%
# def build_generator(PARAMS):
#     z = layers.Input(PARAMS['latent_dim'])
#     y = layers.Input((PARAMS['class_num']))
    
#     hz = layers.Dense(PARAMS['latent_dim'], input_shape = [PARAMS['latent_dim']])(z)
#     hz = layers.LeakyReLU(0.2)(hz)
#     hz = layers.Dense(PARAMS['latent_dim'], input_shape = [PARAMS['latent_dim']])(hz)
#     hz = layers.LeakyReLU(0.2)(hz)
    
#     hy = layers.Dense(PARAMS['latent_dim'])(y)
#     hy = layers.LeakyReLU(0.2)(hy)
    
#     h = layers.Dense(512)(layers.Concatenate()([hz, hy]))
#     h = layers.LeakyReLU(0.2)(h)
#     h = layers.Dense(1024)(layers.Concatenate()([h, hy]))
#     h = layers.LeakyReLU(0.2)(h)
#     h = layers.Dense(3072, activation='sigmoid')(layers.Concatenate()([h, hy]))
#     h = tf.reshape(h, [-1, PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']])
    
#     # h = tf.reshape(h, [-1, 8, 8, 32])
#     # h = layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same')(h)
#     # h = layers.LeakyReLU(0.2)(h)
#     # h = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(h)
#     # h = layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same')(h)
#     # h = layers.LeakyReLU(0.2)(h)
#     # h = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(h)
    
#     # h = layers.Conv2D(filters = 3, kernel_size = 1, activation='sigmoid', padding = 'same')(h)
    
#     G = K.models.Model([z, y], h)
#     # G.summary()
    
#     return G
#%%
def build_image_discriminator(PARAMS):
    x = layers.Input([PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']])

    # h = layers.Conv2D(filters = 8, kernel_size = 3, strides = 2, padding = 'same')(x)
    # h = layers.LeakyReLU(0.2)(h)
    
    h = layers.Flatten()(x)
    h = layers.Dense(1024)(h)
    h = layers.LeakyReLU(0.2)(h)
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

    # h = layers.Conv2D(filters = 8, kernel_size = 3, strides = 2, padding = 'same')(x)
    # h = layers.LeakyReLU(0.2)(h)
    
    h = layers.Flatten()(x)
    h = layers.Dense(1024)(h)
    h = layers.LeakyReLU(0.2)(h)
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