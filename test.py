import keras
from keras.optimizers import Adam
from sonar_models import GAN, discriminator, generator
from fish_dataset import load_dataset2
from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import cv2
from PIL import Image

dis = discriminator()
gen = generator()
Gan = GAN(gen,dis)

Gan.load_weights("./result_sonar/1025/model/gan_weights_lambda100.0_epoch500.h5")
gen = generator()
Gan = GAN(gen,dis)

gen = Gan.layers[2]
dis = Gan.layers[3]

n = 1145
data_ind = np.arange(n)
test_img, test_slabel, test_clabel = load_dataset2(dataDir = "./dataset/train_data/" ,data_range = data_ind[int(n*0.7):],skip = False)
batch_size = 10

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('./test_video.mp4', fourcc,10.0,(256,512))
print(test_slabel.shape)
for i in range(int(test_slabel.shape[0]/batch_size)):
    generated = gen.predict([test_slabel[i*batch_size:min((i+1)*batch_size,n)],test_clabel[i*batch_size:min((i+1)*batch_size,n)]])
    generated = (generated * 128.0 + 128.0).astype(np.uint8)
    print(generated.shape)
    for j in range(generated.shape[0]):
        Image.fromarray(generated[j]).save("./test/test" + str(i*batch_size + j) + ".png")
        video.write(generated[j])

video.release()
   
