#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
# tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
os.chdir('D:\gan')
# os.chdir('/Users/anseunghwan/Documentss/GitHub/gan')

from modules import model_cmnist_vae
#%%
PARAMS = {
    "batch_size": 128,
    "epochs": 10000, 
    "learning_rate": 0.001, 
    "data": "cmnist",
    "class_num": 10,
    "latent_dim": 256, 
    "sigma": 1.0,
}

print(PARAMS)
#%%
if PARAMS['data'] == "cmnist":
    (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
    
    '''colored mnist'''
    PARAMS['data_dim'] = 32
    PARAMS['n_layer'] = int(np.log(PARAMS['data_dim']) / np.log(2)) - 1
    PARAMS["channel"] = 3

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    from tensorflow.keras.utils import to_categorical
    y_train_onehot = to_categorical(y_train, num_classes=PARAMS['class_num'])
    y_test_onehot = to_categorical(y_test, num_classes=PARAMS['class_num'])

    np.random.seed(1)
    def colored_mnist(image):
        image = tf.image.resize(image, [PARAMS['data_dim'], PARAMS['data_dim']], method='nearest')
        
        if tf.random.uniform((1, 1)) > 0.5:
        
            # color
            image0 = tf.cast(image, tf.float32) / 255.
            color = np.random.uniform(0., 1., 3)
            color = color / np.linalg.norm(color)
            image0 = image0 * color[tf.newaxis, tf.newaxis, :]
            
            image0 = (image0.numpy() * 2.) - 1.
            
            assert image0.shape == (PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel'])
            
            return tf.cast(image0, tf.float32)

        else:
            
            # edge detection
            image = cv2.Canny(image.numpy(), 10., 255.)
            image[np.where(image > 0)] = 1.
            image[np.where(image <= 0)] = 0.
            # plt.imshow(laplacian)

            # # width
            # if tf.random.uniform((1, 1)) > 0.5:
            #     kernel = np.ones((2, 2))
            #     image = cv2.dilate(image, kernel)
            # dilation_size = np.random.choice(np.arange(2), 1)[0]
            # kernel = np.ones((dilation_size, dilation_size))
            # image = cv2.dilate(image, kernel)
            # plt.imshow(laplacian)

            # color
            color = np.random.uniform(0., 1., 3)
            color = color / np.linalg.norm(color)
            image = image[..., tf.newaxis] * color[tf.newaxis, tf.newaxis, :]
            # plt.imshow(laplacian)

            # # size
            # size = np.random.uniform(0.4, 0.5, 1)
            # downsize = int(PARAMS['data_dim'] * size) * 2
            # image = tf.image.resize(image, [downsize, downsize], method='nearest')
            # padding = int((PARAMS['data_dim'] - downsize) / 2)
            # image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'constant') 
            # # plt.imshow(laplacian)
            
            # scaling
            image = (image * 2.) - 1.
            
            assert image.shape == (PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel'])
            
            return tf.cast(image, tf.float32)

    cx_train = []
    for i in tqdm(range(len(x_train)), desc='generating colored mnist'):
        cx_train.append(colored_mnist(x_train[i]))
    cx_train = np.array(cx_train)
    # plt.imshow(cx_train[0])
    
    fig, axes = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(40):
        axes.flatten()[i].imshow((cx_train[i] + 1.) / 2.)
        axes.flatten()[i].axis('off')
    plt.savefig('./assets/data_examples.png',
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    # plt.show()
    plt.close()
    
    # np.repeat(y_train_onehot, 2, axis=0)
    train_dataset = tf.data.Dataset.from_tensor_slices((cx_train, y_train_onehot)).shuffle(len(cx_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
else:
    print('Invalid data type!')
    assert 0 == 1
#%%
model = model_cmnist_vae.VAE(PARAMS) 
learning_rate = PARAMS["learning_rate"]
optimizer = K.optimizers.Adam(learning_rate)
lambda_ = 0.7

@tf.function
def train_step(x_batch, y_batch, PARAMS):
    
    with tf.GradientTape() as tape:
        mean, logvar, z, xhat, prob_x, prob_gen = model(x_batch, y_batch, training=True)
        
        loss_, recon_loss, kl_loss = model_cmnist_vae.loss_function(xhat, x_batch, mean, logvar, PARAMS, lambda_) 
        
        cce_x = - tf.reduce_mean(tf.reduce_sum(tf.multiply(y_batch, tf.math.log(prob_x + 1e-8)), axis=-1))
        cce_gen = - tf.reduce_mean(tf.reduce_sum(tf.multiply(y_batch, tf.math.log(prob_gen + 1e-8)), axis=-1))
        
        loss = loss_ + cce_x + 2 * cce_gen
        
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    
    return [loss, loss_, recon_loss, kl_loss, cce_x, cce_gen], [mean, logvar, z, xhat, prob_x, prob_gen]
#%%
def generate_and_save_images(images, epochs):
    fig = plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow((images[i] + 1.) / 2.)
        plt.axis('off')

    plt.savefig('./assets/image_at_epoch_{}.png'.format(epochs))
    # plt.show()
    plt.close()
#%%
step = 0
progress_bar = tqdm(range(PARAMS['epochs']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['epochs']))
#%%
'''training'''
for _ in progress_bar:
    x_batch, y_batch = next(iter(train_dataset))
    step += 1
    
    [loss, loss_, recon_loss, kl_loss, cce_x, cce_gen], [mean, logvar, z, xhat, prob_x, prob_gen] = train_step(x_batch, y_batch, PARAMS)
    
    progress_bar.set_description('setting: {} | iteration {}/{} | loss {:.3f}, recon {:.3f}, kl {:.3f}, cls_x {:.3f}, cls_gen {:.3f}'.format(
        PARAMS['data'], 
        step, PARAMS['epochs'], 
        loss.numpy(), recon_loss.numpy(), kl_loss.numpy(), cce_x.numpy(), cce_gen.numpy()
    ))

    if step % 100 == 0:
        generate_and_save_images(xhat, step)

    if step == PARAMS['epochs']: break
#%%
model.save_weights('./assets/weights_vae/weights')
#%%
model = model_cmnist_vae.VAE(PARAMS) 
model.load_weights('./assets/weights_vae/weights')
#%%
x_batch, y_batch = next(iter(train_dataset))
#%%
mean, logvar, z, xhat, prob_x, prob_gen = model(x_batch, y_batch, training=False)
label = np.zeros((1, PARAMS['class_num']))
label[0, 0] = 1
xhat_ = model.Decoder([z[2][tf.newaxis], label])
#%%
fig = plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.imshow((x_batch[2] + 1.) / 2.)
plt.title('original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow((xhat[2] + 1.) / 2.)
plt.title('reconstruction')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow((xhat_[0] + 1.) / 2.)
plt.title('style transfer given label 0')
plt.axis('off')

plt.savefig('./assets/style_transfer.png')
# plt.show()
plt.close()
#%%
tf.random.set_seed(520)
epsilon = tf.random.normal(shape=(100, PARAMS['latent_dim']))
label = np.zeros((100, PARAMS['class_num']))
label[:, 8] = 1
xhat = model.Decoder([epsilon, label])

fig = plt.figure(figsize=(10, 10))

for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow((xhat[i] + 1.) / 2.)
    plt.axis('off')
plt.suptitle('with label 8', y=0.9)
plt.savefig('./assets/generated.png', bbox_inches="tight", pad_inches=0.1)
plt.show()
# plt.close()
#%%