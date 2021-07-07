#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class Upsampling(layers.Layer):
    def __init__(self, in_ch, with_conv):
        super(Upsampling, self).__init__()
        
        self.in_ch = in_ch
        self.with_conv = with_conv
        self.conv = layers.Conv2D(filters=self.in_ch, kernel_size=3, strides=1, padding='same', name='conv_up')

    def call(self, x, **kwargs):
        B, H, W, C = x.shape
        x = tf.image.resize(x, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # assert x.shape == [B, H * 2, W * 2, C]
        if self.with_conv:
            x = self.conv(x)
            # assert x.shape == [B, H * 2, W * 2, C]
        return x    
#%%
class Downsampling(layers.Layer):
    def __init__(self, in_ch, with_conv):
        super(Downsampling, self).__init__()
        
        self.in_ch = in_ch
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = layers.Conv2D(filters=self.in_ch, kernel_size=3, strides=2, padding='same', name='conv_down')
        else:
            self.avgpool = layers.AveragePooling2D(pool_size=(2, 2), strides=2)

    def call(self, x, **kwargs):
        # B, H, W, C = x.shape
        if self.with_conv:
            x = self.conv(x)
        else:
            x = self.avgpool(x)
        # assert x.shape == [B, H // 2, W // 2, C]
        return x
#%%
class AttentionBlock(layers.Layer):
    def __init__(self, in_ch):
        super(AttentionBlock, self).__init__()
        
        self.in_ch = in_ch
        self.normalize = layers.LayerNormalization()
        self.q_layer = layers.Dense(self.in_ch, name='q')
        self.k_layer = layers.Dense(self.in_ch, name='k')
        self.v_layer = layers.Dense(self.in_ch, name='v')
        
        self.proj_out = layers.Dense(self.in_ch, name='proj_out')

    def call(self, x, **kwargs):
        B, H, W, C = x.shape
        h = self.normalize(x)
        q = self.q_layer(h)
        k = self.k_layer(h)
        v = self.v_layer(h)
        
        w = tf.einsum('bhwc,bHWc->bhwHW', q, k) * (int(self.in_ch) ** (-0.5))
        w = tf.reshape(w, [-1, H, W, H * W])
        w = tf.nn.softmax(w, -1)
        w = tf.reshape(w, [-1, H, W, H, W])
        
        h = tf.einsum('bhwHW,bHWc->bhwc', w, v)
        h = self.proj_out(h)
        
        # assert h.shape == x.shape
        return x + h
#%%
class ResnetBlock(layers.Layer):
    def __init__(self, in_ch, out_ch=None):
        super(ResnetBlock, self).__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        if self.out_ch is None:
            self.out_ch = self.in_ch
        
        if self.out_ch != self.in_ch:
            self.shortcut = layers.Conv2D(filters=self.out_ch, kernel_size=3, strides=1, padding='same', name='conv_shortcut')
        
        self.normalize1 = layers.LayerNormalization()
        self.normalize2 = layers.LayerNormalization()
        self.conv1 = layers.Conv2D(filters=self.out_ch, kernel_size=3, strides=1, padding='same', name='conv1')
        self.conv2 = layers.Conv2D(filters=self.out_ch, kernel_size=3, strides=1, padding='same', name='conv2')

    def call(self, x, **kwargs):
        h = x
        h = tf.nn.swish(self.normalize1(h))
        h = self.conv1(h)

        h = tf.nn.swish(self.normalize2(h))
        h = self.conv2(h)
        
        if self.out_ch != self.in_ch:
            x = self.shortcut(x)

        # assert x.shape == h.shape
        return x + h
#%%
def build_encoder(PARAMS):
    x = layers.Input((PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']))
    noise = layers.Input((PARAMS['latent_dim']))
    
    embedding_dim = PARAMS['embedding_dim']
    embedding_dim_mult = PARAMS['embedding_dim_mult']
    num_res_blocks = PARAMS['num_res_blocks']
    attn_resolutions = PARAMS['attn_resolutions']
    
    num_resolutions = len(embedding_dim_mult)

    h = layers.Conv2D(filters=embedding_dim, kernel_size=3, strides=1, padding='same', name='conv_in')(x)
    for i_level in range(num_resolutions):
        # Residual blocks for this resolution
        for i_block in range(num_res_blocks):
            h = ResnetBlock(in_ch=h.shape[-1], out_ch=embedding_dim * embedding_dim_mult[i_level])(h)
            if h.shape[1] in attn_resolutions:
                h = AttentionBlock(in_ch=h.shape[-1])(h)
        # Downsample
        if i_level != num_resolutions - 1:
            h = Downsampling(in_ch=h.shape[-1], with_conv=True)(h)
    
    h = layers.Flatten()(h)
    h = layers.Dense(PARAMS['latent_dim'])(h)
    
    noise_h = layers.Dense(PARAMS['latent_dim'])(noise)
    h = h * noise_h
    
    E = K.models.Model([x, noise], h)
    # E.summary()
    
    return E
#%%
def upsample(x):
    return tf.image.resize(x, [2, 2], method='bilinear')
#%%
def crop_to_fit(x):
    height = x[1].shape[1]
    width = x[1].shape[2]
    return x[0][:, :height, :width, :]
#%%
def AdaIN(x):
    mean, std = tf.nn.moments(x[0], axes=[1, 2], keepdims=True)
    normalized = (x[0] - mean) / (std + 1e-8)

    pool_shape = [-1, 1, 1, normalized.shape[-1]]
    gamma = tf.reshape(x[1], pool_shape)
    beta = tf.reshape(x[2], pool_shape)
    return normalized * gamma + beta
#%%
def gen_block(x, style, noise, filters, up=True):
    if up:
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

    # A
    gamma = layers.Dense(filters)(style)
    beta = layers.Dense(filters)(style)
    # B
    delta = layers.Lambda(crop_to_fit)([noise, x])
    delta = layers.Dense(filters, kernel_initializer = 'zeros')(delta)

    h = layers.Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    h = layers.add([h, delta])
    h = layers.Lambda(AdaIN)([h, gamma, beta])
    h = layers.LeakyReLU(0.2)(h)
    return h
#%%
def build_styler(PARAMS):
    # === Style Mapping ===

    S = K.models.Sequential()

    S.add(layers.Dense(PARAMS['latent_dim'], input_shape = [PARAMS['latent_dim']]))
    S.add(layers.LeakyReLU(0.2))
    S.add(layers.Dense(PARAMS['latent_dim']))
    S.add(layers.LeakyReLU(0.2))
    S.add(layers.Dense(PARAMS['latent_dim']))
    S.add(layers.LeakyReLU(0.2))
    S.add(layers.Dense(PARAMS['latent_dim']))
    S.add(layers.LeakyReLU(0.2))
    
    return S
#%%
def build_synthesis(PARAMS):
    # === synthesis network ===

    n_layers = PARAMS['n_layer']

    input_style = []
    for i in range(n_layers):
        input_style.append(layers.Input([PARAMS['latent_dim']]))

    input_noise = layers.Input([PARAMS['data_dim'], PARAMS['data_dim'], 1])

    y = layers.Input((PARAMS['class_num']))

    dim = 32
    # Actual Model
    x = layers.Dense(4*4*4*dim, activation = 'relu', kernel_initializer = 'he_normal')(y)
    x = layers.Reshape([4, 4, 4*dim])(x)
    x = gen_block(x, input_style[0], input_noise, 16 * dim, up = False)  
    x = gen_block(x, input_style[1], input_noise, 8 * dim)  
    x = gen_block(x, input_style[2], input_noise, 6 * dim)  
    x = gen_block(x, input_style[3], input_noise, 4 * dim)  

    x = layers.Conv2D(filters = 3, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = tf.nn.sigmoid(x)

    G = K.models.Model(inputs = input_style + [input_noise] + [y], outputs = x)
    # G.summary()
    
    return G
#%%
def build_generator(PARAMS):
    # === Generator ===
    
    n_layers = PARAMS['n_layer']
    
    S = build_styler(PARAMS)
    G = build_synthesis(PARAMS)
    
    input_style = []
    style = []
    y = layers.Input((PARAMS['class_num']))

    for i in range(n_layers):
        input_style.append(layers.Input([PARAMS['latent_dim']]))
        style.append(S(input_style[-1]))

    input_noise = layers.Input([PARAMS['data_dim'], PARAMS['data_dim'], 1])

    G_output = G(style + [input_noise] + [y])

    GM = K.models.Model(inputs = input_style + [input_noise] + [y], outputs = G_output)

    return GM
#%%
def dis_block(x, filters, pooling = True):

    h = layers.Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    h = layers.LeakyReLU(0.2)(h)

    if pooling:
        h = layers.AveragePooling2D()(h)

    return h
#%%
def build_image_discriminator(PARAMS):
    x = layers.Input([PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']])

    dim = 16

    h = dis_block(x, 1 * dim)  
    h = dis_block(h, 2 * dim) 
    h = dis_block(h, 4 * dim)
    h = dis_block(h, 8 * dim) 
    h = dis_block(h, 16 * dim) 

    h = layers.Flatten()(h)

    h = layers.Dense(4 * dim, kernel_initializer = 'he_normal')(h)
    h = layers.LeakyReLU(0.2)(h)

    h = layers.Dense(1, kernel_initializer = 'he_normal', activation='sigmoid')(h)

    D = K.models.Model(inputs = x, outputs = h)
    # D.summary()

    return D
#%%
def build_image_classifier(PARAMS):
    x = layers.Input([PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']])

    dim = 16

    h = dis_block(x, 1 * dim)  
    h = dis_block(h, 2 * dim) 
    h = dis_block(h, 4 * dim)
    h = dis_block(h, 8 * dim) 
    h = dis_block(h, 16 * dim) 

    h = layers.Flatten()(h)

    h = layers.Dense(4 * dim, kernel_initializer = 'he_normal')(h)
    h = layers.LeakyReLU(0.2)(h)

    h = layers.Dense(PARAMS['class_num'], kernel_initializer = 'he_normal', activation='softmax')(h)

    D = K.models.Model(inputs = x, outputs = h)
    # D.summary()

    return D
#%%
def build_z_discriminator(PARAMS):
    x = layers.Input([PARAMS['latent_dim']])

    h = layers.Dense(128, kernel_initializer = 'he_normal')(x)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(64, kernel_initializer = 'he_normal')(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(32, kernel_initializer = 'he_normal')(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(16, kernel_initializer = 'he_normal')(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Dense(8, kernel_initializer = 'he_normal')(h)
    h = layers.LeakyReLU(0.2)(h)

    h = layers.Dense(1, kernel_initializer = 'he_normal', activation='sigmoid')(h)

    D = K.models.Model(inputs = x, outputs = h)
    # D.summary()

    return D
#%%