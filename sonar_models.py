import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense,Reshape, Input,merge
from keras.layers.merge import concatenate
from keras.layers.core import Activation, Dropout, Flatten,Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D,Conv2D, MaxPooling2D,Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU


# convolution batchnormalization relu
def CBR(ch,shape,bn=True,sample='down',activation=LeakyReLU, dropout=False):
    model = Sequential()
    if sample=='down':
        model.add(Conv2D(filters=ch, kernel_size=(4,4), strides=2, padding='same',input_shape=shape))
    else:
        model.add(Conv2DTranspose(filters=ch, kernel_size=(4,4), strides=2, padding='same',input_shape=shape))
    if bn:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(0.5))
    if activation == LeakyReLU:
        model.add(LeakyReLU(alpha=0.2))
    else:
        model.add(Activation('relu'))

    return model



def discriminator():
    h = 512
    w = 256
    img = Input(shape=(h,w,3))
    x = CBR(32,(512,256,3),bn=False)(img)
    x = CBR(64,(256,128,16))(x)
    x = CBR(128,(128,64,32))(x)
    x = CBR(256,(64,32,64))(x)
    x = CBR(512,(32,16,128))(x)

    x = Conv2D(filters=1,kernel_size=3,strides=1,padding='same')(x)
    x = Activation('sigmoid')(x)
    output = Lambda(lambda x: K.mean(x, axis=[1,2]),output_shape=(1,))(x)
    model = Model(inputs =[img], outputs = [output])

    return model


def generator():

    # encoder
    input1 = Input(shape=(512,256,1))
    x = Conv2D(filters=8, kernel_size=(3,3), strides=1, padding='same',input_shape=(512,256,3))(input1)
    x = CBR(16,(512,256,8))(x)
    x = CBR(32,(256,128,16))(x)
    x = CBR(64,(128,64,32))(x)
    x = CBR(128,(64,32,64))(x)
    x = CBR(256,(32,16,128))(x)
    x = CBR(512,(16,8,256))(x)
    x = Conv2D(filters=512, kernel_size=(8,4), strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x1 = LeakyReLU(0.2)(x)

    input2 = Input(shape=(512,256,3))
    x = Conv2D(filters=8, kernel_size=(3,3), strides=1, padding='same',input_shape=(512,256,1))(input2)
    x = CBR(16,(512,256,8))(x)
    x = CBR(32,(256,128,16))(x)
    x = CBR(64,(128,64,32))(x)
    x = CBR(128,(64,32,64))(x)
    x = CBR(256,(32,16,128))(x)
    x = CBR(512,(16,8,256))(x)
    x = Conv2D(filters=512, kernel_size=(8,4), strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x2 = LeakyReLU(0.2)(x)

    x = concatenate([x1,x2])
    x = Dense(1024*2*1)(x)
    x = Reshape((2,1,1024))(x)

    # decoder
    x = CBR(512,(2,1,1024),sample='up',activation='relu',dropout=True)(x)
    x = CBR(512,(4,2,512),sample='up',activation='relu',dropout=True)(x)
    x = CBR(256,(8,4,512),sample='up',activation='relu',dropout=True)(x)
    x = CBR(128,(16,8,256),sample='up',activation='relu',dropout=False)(x)
    x = CBR(64,(32,16,128),sample='up',activation='relu',dropout=False)(x)
    x = CBR(32,(64,32,64),sample='up',activation='relu',dropout=False)(x)
    x = CBR(16,(128,64,32),sample='up',activation='relu',dropout=False)(x)
    x = CBR(8,(128,64,16),sample='up',activation='relu',dropout=False)(x)
    output = Conv2D(filters=3, kernel_size=(3,3),strides=1,padding="same")(x)

    model = Model(inputs=[input1,input2], outputs=output)
    return(model)


def GAN(generator, discriminator):

    s_input = Input(shape=(512,256,1))
    c_input = Input(shape=(512,256,3))
    generated_image = generator([s_input,c_input])
    DCGAN_output = discriminator(generated_image)
    DCGAN = Model(inputs=[s_input,c_input],outputs=[generated_image, DCGAN_output],name="DCGAN")

    return DCGAN
