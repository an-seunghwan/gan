#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
#%%
def build_encoder(PARAMS):
    x = layers.Input((PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']))
    
    dims = [8, 16, 32, 64]
    skip = x
    for i in range(PARAMS['n_layer']):
        skip = layers.Conv2D(filters = dims[i], kernel_size = 3, strides = 2, padding = 'same')(skip)
        skip = layers.BatchNormalization()(skip)
        skip = layers.LeakyReLU(0.2)(skip)
        
        h = layers.Conv2D(filters = dims[i], kernel_size = 3, strides = 1, padding = 'same')(skip)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU(0.2)(h)
        
        h = layers.Conv2D(filters = dims[i], kernel_size = 3, strides = 1, padding = 'same')(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU(0.2)(h)
        
        skip = h + skip
    
    mean = layers.Dense(PARAMS['latent_dim'])(layers.Flatten()(skip))
    logvar = layers.Dense(PARAMS['latent_dim'])(layers.Flatten()(skip))
    
    E = K.models.Model(x, [mean, logvar])
    E.summary()
    
    return E
#%%
def build_generator(PARAMS):
    z = layers.Input(PARAMS['latent_dim'])
    y = layers.Input(PARAMS['class_num'])
    
    hy = layers.Dense(4, use_bias=False)(y)[..., tf.newaxis]
    hy = tf.matmul(hy, hy, transpose_b=True)[..., tf.newaxis]
    
    h = layers.Reshape((4, 4, 16))(z)
    h = layers.Concatenate()([h, hy])
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU(0.2)(h)
    
    dims = [128, 64, 32]
    skip = h
    for i in range(3):
        skip = layers.Conv2DTranspose(filters = dims[i], kernel_size = 5, strides = 2, padding = 'same', use_bias=False)(skip)
        skip = layers.BatchNormalization()(skip)
        skip = layers.LeakyReLU(0.2)(skip)
        
        h = layers.Conv2D(filters = dims[i], kernel_size = 5, strides = 1, padding = 'same', use_bias=False)(skip)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU(0.2)(h)
        
        h = layers.Conv2D(filters = dims[i], kernel_size = 5, strides = 1, padding = 'same', use_bias=False)(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU(0.2)(h)
        
        skip = h + skip
    
    h = layers.Conv2DTranspose(3, (5, 5), strides=1, padding='same', use_bias=False, activation='tanh')(skip)
    
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
def build_discriminator(PARAMS):
    x = layers.Input([PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']])

    h = layers.Conv2D(filters = 64, kernel_size = 5, strides = 2, padding = 'same')(x)
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dropout(0.3)(h)
    
    h = layers.Conv2D(filters = 128, kernel_size = 5, strides = 2, padding = 'same')(h)
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dropout(0.3)(h)
    
    h = layers.Conv2D(filters = 256, kernel_size = 5, strides = 2, padding = 'same')(h)
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dropout(0.3)(h)
    
    h = layers.Flatten()(h)
    
    dis = layers.Dense(1, activation='sigmoid')(h)
    cls = layers.Dense(PARAMS['class_num'], activation='softmax')(h)

    D = K.models.Model(inputs = x, outputs = [dis, cls])
    D.summary()

    return D
#%%
# def build_image_classifier(PARAMS):
#     x = layers.Input([PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']])

#     # h = layers.Conv2D(filters = 8, kernel_size = 3, strides = 2, padding = 'same')(x)
#     # h = layers.LeakyReLU(0.2)(h)
    
#     h = layers.Flatten()(x)
#     h = layers.Dense(1024)(h)
#     h = layers.LeakyReLU(0.2)(h)
#     h = layers.Dense(512)(h)
#     h = layers.LeakyReLU(0.2)(h)
#     h = layers.Dense(128)(h)
#     h = layers.LeakyReLU(0.2)(h)

#     h = layers.Dense(PARAMS['class_num'], activation='softmax')(h)

#     D = K.models.Model(inputs = x, outputs = h)
#     # D.summary()

#     return D
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