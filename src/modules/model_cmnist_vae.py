#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
#%%
class ConvLayer(K.layers.Layer):
    def __init__(self, filter_size, kernel_size, strides, name="ConvLayer", **kwargs):
        super(ConvLayer, self).__init__(name=name, **kwargs)
        self.conv2d = layers.Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, padding='same')
        self.batchnorm = layers.BatchNormalization()

    def call(self, x):
        h = self.conv2d(x)
        h = tf.nn.relu(h)
        h = self.batchnorm(h)
        return h
#%%
class DeConvLayer(K.layers.Layer):
    def __init__(self, filter_size, kernel_size, strides, name="DeConvLayer", **kwargs):
        super(DeConvLayer, self).__init__(name=name, **kwargs)
        self.deconv2d = layers.Conv2DTranspose(filters=filter_size, kernel_size=kernel_size, strides=strides, padding='same')
        self.batchnorm = layers.BatchNormalization()

    def call(self, x):
        h = self.deconv2d(x)
        h = tf.nn.relu(h)
        h = self.batchnorm(h)
        return h
#%%
class Encoder(K.models.Model):
    def __init__(self, params, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.encoder_conv1 = ConvLayer(32, 5, 2)
        self.encoder_conv2 = ConvLayer(64, 5, 2)
        self.encoder_conv3 = ConvLayer(128, 3, 2)
        self.encoder_conv4 = ConvLayer(256, 3, 2)
        self.encoder_dense = layers.Dense(1024, activation='relu')
        self.batchnorm = layers.BatchNormalization()
        
    def call(self, x):
        h = self.encoder_conv1(x)
        h = self.encoder_conv2(h)
        h = self.encoder_conv3(h)
        h = self.encoder_conv4(h)
        h = layers.Flatten()(h)
        h = self.batchnorm(self.encoder_dense(h))
        return h
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
        skip = layers.ReLU()(skip)
        
        h = layers.Conv2D(filters = dims[i], kernel_size = 5, strides = 1, padding = 'same', use_bias=False)(skip)
        h = layers.BatchNormalization()(h)
        h = layers.ReLU()(h)
        
        h = layers.Conv2D(filters = dims[i], kernel_size = 5, strides = 1, padding = 'same', use_bias=False)(h)
        h = layers.BatchNormalization()(h)
        h = layers.ReLU()(h)
        
        skip = h + skip
    
    h = layers.Conv2DTranspose(3, (1, 1), strides=1, padding='same', use_bias=False, activation='tanh')(skip)
    
    G = K.models.Model([z, y], h)
    # G.summary()
    
    return G
#%%
class Classifier(K.models.Model):
    def __init__(self, params, name="Classifier", **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.classifier_cnn1 = layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same', use_bias=False)
        self.classifier_cnn2 = layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same', use_bias=False)
        self.classifier_pool1 = layers.MaxPooling2D((2, 2))
        self.classifier_cnn3 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same', use_bias=False)
        self.classifier_cnn4 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same', use_bias=False)
        self.classifier_pool2 = layers.MaxPooling2D((2, 2))
        self.classifier_cnn5 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same', use_bias=False)
        self.classifier_cnn6 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same', use_bias=False)
        self.classifier_pool3 = layers.MaxPooling2D((2, 2))
        self.classifier_dense = layers.Dense(128, activation='relu')
        self.classifier_logit = layers.Dense(self.params['class_num'], activation='softmax', use_bias=False)

    def call(self, x):
        h = self.classifier_pool1(self.classifier_cnn2(self.classifier_cnn1(x)))
        h = self.classifier_pool2(self.classifier_cnn4(self.classifier_cnn3(h)))
        h = self.classifier_pool3(self.classifier_cnn6(self.classifier_cnn5(h)))
        h = layers.Flatten()(h)
        h = self.classifier_dense(h)
        h = self.classifier_logit(h)
        return h
#%%
class VAE(K.models.Model):
    def __init__(self, params, name='VAE', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.params = params
        
        self.Encoder = Encoder(self.params)
        
        self.mean_layer = layers.Dense(self.params['latent_dim'], activation='linear')
        self.logvar_layer = layers.Dense(self.params['latent_dim'], activation='linear')
        
        # discrete latent (eta)
        self.classifier = Classifier(self.params)

        # decoder (theta)
        self.Decoder = build_generator(self.params)
        
    def encoder(self, x):
        h = self.Encoder(x)
        return h
    
    def decoder(self, x):
        h = self.Decoder(x)
        return h
    
    def call(self, x, y):
        latent_dim = self.params["latent_dim"]   

        h = self.encoder(x)
        
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        epsilon = tf.random.normal((tf.shape(x)[0], latent_dim))
        z = mean + tf.math.exp(logvar / 2) * epsilon 
        # assert z.shape == (tf.shape(x)[0], class_num, latent_dim)
        
        xhat = self.decoder([z, y]) 
        # assert xhat.shape == (tf.shape(x)[0], self.params["data_dim"], self.params["data_dim"], 3)
        
        prob_x = self.classifier(x)
        prob_gen = self.classifier(xhat)
        
        return mean, logvar, z, xhat, prob_x, prob_gen
#%%
def loss_function(xhat, x, mean, logvar, PARAMS, lambda_):
    # reconstruction with Laplacian
    error1 = lambda_ * tf.reduce_mean(tf.reduce_sum(tf.abs(xhat - x), axis=[1,2,3]))
    error2 = (1 - lambda_) * tf.reduce_mean(tf.reduce_sum(tf.square(xhat - x), axis=[1,2,3]))
    error = error1 + error2

    kl_error = tf.reduce_mean(0.5 * (tf.reduce_sum(tf.math.pow(mean, 2) / PARAMS['sigma'], axis=-1) 
                                    - PARAMS['latent_dim']
                                    + tf.reduce_sum(tf.math.log(PARAMS['sigma']))
                                    + tf.reduce_sum(tf.math.exp(logvar) / PARAMS['sigma'], axis=-1)
                                    - tf.reduce_sum(logvar, axis=-1)))
            
    return error + kl_error, error, kl_error
#%%