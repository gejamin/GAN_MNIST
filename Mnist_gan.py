import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

(train_x,train_y),(test_x,test_y)  = keras.datasets.mnist.load_data()
train_x=train_x.astype('float32')
train_x=train_x/255.
test_x=test_x.astype('float32')
test_x=test_x/255.
train_db=tf.data.Dataset.from_tensor_slices((train_x,train_y)).shuffle(60000).batch(256,drop_remainder=True)
test_db=tf.data.Dataset.from_tensor_slices((test_x,test_y)).batch(256)

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1=keras.layers.Dense(64)
        self.leaky_rely1=keras.layers.LeakyReLU(alpha=0.01)
        self.fc2 = keras.layers.Dense(128)
        self.leaky_rely2 = keras.layers.LeakyReLU(alpha=0.01)
        self.fc3 = keras.layers.Dense(512)
        self.leaky_rely3 = keras.layers.LeakyReLU(alpha=0.01)
        self.fc4=keras.layers.Dense(784)
    def call(self, inputs, trainig=None):
        x = inputs
        x = self.leaky_rely1(self.fc1(x))
        x = self.leaky_rely2(self.fc2(x))
        x = self.leaky_rely3(self.fc3(x))
        x = self.fc4(x)
        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = keras.layers.Dense(512)
        self.leaky1=keras.layers.LeakyReLU(alpha=0.01)
        self.fc2 = keras.layers.Dense(256)
        self.leaky2 = keras.layers.LeakyReLU(alpha=0.01)
        self.fc3 = keras.layers.Dense(64)
        self.leaky3 = keras.layers.LeakyReLU(alpha=0.01)
        self.fc4=keras.layers.Dense(1)
    def call(self,inputs,training=None):
        x = inputs
        x = self.leaky1(self.fc1(x))
        x = self.leaky2(self.fc2(x))
        x = self.leaky3(self.fc3(x))
        x = self.fc4(x)
        return x

def gen_loss(generator,discriminator,rand,is_training):
    out=generator(rand,is_training)
    logit=discriminator(out,is_training)
    loss=tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(logit),logit)
    return tf.reduce_mean(loss)

def disc_loss(generator,discriminator,rand,true_img,is_training):
    # rand [256, 10]=> [256, 784]
    fals_img=generator(rand,is_training)
    logit=discriminator(fals_img,is_training)
    logit2=discriminator(true_img,is_training)
    #print(logit,logit2)
    loss_g=tf.losses.binary_crossentropy(tf.zeros_like(logit),logit,from_logits=True,)
    loss_d=tf.losses.binary_crossentropy(tf.ones_like(logit2),logit2,from_logits=True)
    return tf.reduce_mean(loss_d+loss_g)

def main():
    gen=Generator()
    disc=Discriminator()

    # print(gen(tf.random.normal([256,10])).shape)
    # print(disc(tf.random.normal([256,784])).shape)
    # loss=disc_loss(gen,disc,tf.random.normal([256,10]),tf.random.normal([256,784]))
    # print(loss)
    optimizer_d=keras.optimizers.RMSprop()
    optimizer_g=keras.optimizers.RMSprop()
    for epoch in range(20):
        for step, (x,_) in enumerate(train_db):
            rand = tf.random.uniform([256,10],-1.,1.)
            x=tf.reshape(x,[-1,784])
            #训练Discriminator
            with tf.GradientTape() as tape:
                loss_d=disc_loss(gen,disc,rand,x,is_training=True)
            grads_d=tape.gradient(loss_d,disc.trainable_variables)
            optimizer_d.apply_gradients(zip(grads_d,disc.trainable_variables))
            #训练Generator
            with tf.GradientTape() as tape:
                loss_g = gen_loss(gen, disc, rand,is_training=True)
            grads_g=tape.gradient(loss_g,gen.trainable_variables)
            optimizer_g.apply_gradients(zip(grads_g,gen.trainable_variables))
            if step % 100 == 0:
                print(step,loss_d,loss_g)
                rand_test=tf.random.uniform([1,10],minval=-1.,maxval=1.)
                img=gen(rand_test)
                print(tf.reshape(img,[28,28]).shape)
                plt.imsave('./{}_{}.png'.format(epoch,step),tf.reshape(img,[28,28])*255.,cmap='gray')

if __name__ == '__main__':
    main()


