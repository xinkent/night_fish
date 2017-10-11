import numpy as np
from PIL import Image

from io import BytesIO


def load_dataset(dataDir='./dataset/train_data/', data_range=range(0,300)):
        print("load dataset start")
        print("     from: %s"%dataDir)
        imgDataset = []
        labelDataset = []

        imgStart = 1515
        labelStart = 12313
        for i in data_range:
            imgNum = imgStart + int(i*(29/10))
            labelNum = labelStart + int(i*(29/10))
            img = Image.open(dataDir + "GP030023_%06d.png"%imgNum)
            label = Image.open(dataDir + "GP029343_%06d.png"%labelNum)
            img = img.resize((512,256), Image.BILINEAR)
            img= img.transpose(Image.ROTATE_90)
            label = label.resize((256, 512),Image.BILINEAR)

            img = np.asarray(img)/128.0-1.0

            label = np.asarray(label)/128.0-1.0

            imgDataset.append(img)
            labelDataset.append(label)

        print("load dataset done")
        return np.array(imgDataset),np.array(labelDataset)

def load_dataset2(dataDir='./dataset/train_data/', data_range=range(0,300)):
        print("load dataset start")
        print("     from: %s"%dataDir)
        imgDataset = []
        clabelDataset = []
        slabelDataset = []

        imgStart = 1515
        slabelStart = 1
        clabelStart = 12313
        for i in data_range:
            imgNum = imgStart + int(i*(29/10))
            slabelNum = slabelStart + i
            clabelNum = clabelStart + int(i*(29/10))
            img = Image.open(dataDir + "GP030023_%06d.png"%imgNum)
            label_sonar = Image.open(dataDir + "2017-03-02_105804_%d_width_773_height_1190.png"%slabelNum)
            label_color = Image.open(dataDir + "GP029343_%06d.png"%clabelNum)

            label_sonar = label_sonar.convert("L")
            img = img.resize((512,256), Image.BILINEAR)
            img= img.transpose(Image.ROTATE_90)
            label_sonar = label_sonar.resize((256, 512),Image.BILINEAR)
            label_color = label_color.resize((256, 512),Image.BILINEAR)

            img = np.asarray(img)/128.0-1.0

            label_sonar = np.asarray(label_sonar)/128.0-1.0
            label_sonar = label_sonar[:,:,np.newaxis]

            label_color = np.asarray(label_color)/128.0-1.0

            imgDataset.append(img)
            slabelDataset.append(label_sonar)
            clabelDataset.append(label_color)


        print("load dataset done")
        return np.array(imgDataset),np.array(slabelDataset),np.array(clabelDataset)
