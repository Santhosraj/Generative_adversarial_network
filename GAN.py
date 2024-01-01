# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 23:16:48 2023

@author: SanthosRaj
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import ipywidgets
import tensorflow_datasets as tfds


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
for gpu in gpus:
    print(gpu)
    
    
    
 
   #using tensorflow  dataset api to brin
ds = tfds.load('fashion_mnist', split = 'train')
#ds.as_numpy_iterator().next().keys()
#ds.as_numpy_iterator().next()['image']
ds.take(2).cache().repeat()

#data transformation 
import numpy as np

data_iterator = ds.as_numpy_iterator()

#getting data out of the pipelineee
np.squeeze(data_iterator.next()['image']).shape

fig,ax = plt.subplots(ncols= 4,figsize =(20,20))
for idx in range(4):
    batch = data_iterator.next()
    ax[idx].imshow(np.squeeze(batch['image']))
    ax[idx].title.set_text(batch['label'])
    
def scale_images(data):
    image = data['image']
    return image/255    




'''
steps for data pipelining :
    map
    cache
    shuffle
    batch
    prefetch
'''

#reload the dataset
ds = tfds.load('fashion_mnist',split ='train')
#running the dataset through the scaled images
ds = ds.map(scale_images)
#cache dataset for that batch
ds = ds.cache()
#shuffle 
ds = ds.shuffle(60000)
#batch into 128 images sample
ds = ds.batch(128)
#reduces the likelihood of the bottleneckingo9
ds = ds.prefetch(64)




#building a neural network

#generator - gives the output that could convince the discriminator 
#discriminator - determines/discriminate if the output is real or fake


from keras.models import Sequential
from keras.layers import Conv2D , Dense, Flatten , Reshape , LeakyReLU , Dropout, UpSampling2D


#build generator

def build_generator():
    #begining of image generation
    model = Sequential()
    model .add(Dense(7*7*128,input_dim = 128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))
    
    #upsampling block 1
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5,padding='same'))
    model.add(LeakyReLU(0.2))
    
    #upsampling block2
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5,padding='same'))
    model.add(LeakyReLU(0.2))
    
    #convolutional  block 1
    model.add(Conv2D(128,4, padding='same'))
    model.add(LeakyReLU(0.2))
    
    #convolutional block 2
    model.add(Conv2D(128,4, padding='same'))
    model.add(LeakyReLU(0.2))
    
    #conv layer to get to one channel
    model.add(Conv2D(1,4,padding='same',activation='sigmoid'))
    
    return model

generator = build_generator()
print(generator.summary())
img = generator.predict(np.random.rand(4,128,1))
img.shape

#generate new images
img = generator.predict(np.random.rand(4,128,1))

fig ,  ax = plt.subplots(ncols=4 , figsize = (20,20))

for idx,img in enumerate(img):
   
    ax[idx].imshow(np.squeeze(img))
    ax[idx].title.set_text(idx)
 


#build discriminator

def build_discriminator():
    model = Sequential()
    
    #firstconv block
    model.add(Conv2D(32, 5, input_shape = (28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    #second conv layer
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    
    #third conv layer
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1,activation='sigmoid'))    
    return model
    
    
discriminator= build_discriminator()
discriminator.summary()
img = generator.predict(np.random.rand(4,128,1))

discriminator.predict(img)

#generate new images
img = generator.predict(np.random.rand(4,128,1))

fig ,  ax = plt.subplots(ncols=4 , figsize = (20,20))

for idx,img in enumerate(img):
   
    ax[idx].imshow(np.squeeze(img))
    ax[idx].title.set_text(idx)


#custom training 

 
#training generator and discriminator simultaneously(adding additional noise)

#setup losses and optimizers
from  keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

#building a subclass model
#Importing base model class to subclass
from keras.models import Model

class FashionGAN(Model):
    def __init__(self, generator,discriminator , *args,**kwargs):
        #pass through args and kwargs to base class
        super().__init__(*args,**kwargs)
        
        #create attributes for gen and disc
        self.generator = generator
        self.discriminator = discriminator
        
        
    
    def compile(self,g_opt,d_opt ,g_loss,d_loss,*args,**kwargs):
        #compile with base class
        super().compile(*args,**kwargs)
        
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss
        
        
    
    def train_step(self,batch):
        #get data
        real_images = batch
        fake_images = self.generator(tf.random.normal((128,128,1)),training = False)
        
        #step 1 :Train the discriminator
        with tf.GradientTape() as d_tape:
            yhat_real = self.discriminator(real_images,training = True)
            yhat_fake = self.discriminator(fake_images, training = True)
            yhat_realfake = tf.concat([yhat_real,yhat_fake],axis = 0)
            
            #create labels for real and fake images
            y_realfake = tf.concat([tf.zeros_like(yhat_real),tf.ones_like(yhat_fake)],axis = 0)
            
            
            #add some noise to the outputs
            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            
            y_realfake += tf.concat([noise_real , noise_fake],axis = 0)
            
            #calculate loss
            total_d_loss = self.d_loss(y_realfake , yhat_realfake)
       
        
       #backpropogation
        dgrad = d_tape.gradient(total_d_loss ,self.discriminator.trainable_variables )
        self.d_opt.apply_gradients(zip(dgrad , self.discriminator.trainable_variables))
        
        #train the generator
        with tf.GradientTape() as g_tape:
            #generate some new images
            gen_images = self.generator(tf.random.normal((128,128,1)),training = True)
            
            #create the predicted labels
            predicted_labels = self.discriminator(gen_images,training = False)
            
            #calculate loss - to fake the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels),predicted_labels)
        #apply backprop
        
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad,self.generator.trainable_variables))
        
        return{"d_loss":total_d_loss,"g_loss":total_g_loss}
        
fashgan = FashionGAN(generator, discriminator)
fashgan.compile(g_opt, d_opt, g_loss, d_loss)

#build callback
import os
from keras.utils import array_to_img
from keras.callbacks import Callback
                                     
class ModelMonitor(Callback):
    def __init__(self,num_img = 3 , latent_dim = 128):
        self.num_img = num_img
        self.latent_dim = latent_dim
    
    def on_epoch_end(self,epoch,logs = None):
        random_latent_vectors  = tf.random.uniform((self.num_img,self.latent_dim,1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *=255
        generated_images.numpy()
        for i in range(self.num_img) :
            img = array_to_img(generated_images[i])
            img.save(os.path.join('D:/Santhosraj Machine learning/spyder/GANimages',f'generated_img_{epoch}_{i}.png'))
            
#training the model
hist = fashgan.fit(ds,epochs=20,callbacks=[ModelMonitor()])



#review
plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.show()


#generate images

imgs = generator.predict(tf.random.normal((16, 128, 1)))
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10,10))
for r in range(4): 
    for c in range(4): 
        ax[r][c].imshow(imgs[(r+1)*(c+1)-1])