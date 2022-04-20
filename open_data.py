from numpy import load
import numpy as np
from PIL import Image
from numpy import asarray
import os

'''MNIST
data_test = load('data\moving-mnist-example\moving-mnist-test.npz')
data_train = load('data\moving-mnist-example\moving-mnist-train.npz')
data_valid = load('data\moving-mnist-example\moving-mnist-valid.npz')

print('---------------------Training Data---------------------')
lst = data_train.files
for item in lst:
    print('' + item + ':', np.array(data_train[item]).shape)
    
print('---------------------Testing Data---------------------')
lst = data_test.files
for item in lst:
    print('' + item + ':', np.array(data_test[item]).shape)

print('---------------------Validation Data---------------------')
lst = data_valid.files
for item in lst:
    print('' + item + ':', np.array(data_valid[item]).shape)
'''

'''Read our Data .. for 256'''

def getData():
    scene_256 = ['CA_0020', 'CA_0780', 'CA_0790', 'CA_0800', 'CA_1120', 'CA_1720','CA_1750']
    ims = np.array([])
    for scene in os.listdir("data/Training_crop"):
        sequence = np.array([])
        if scene in scene_256:
            print(scene)
            for image in os.listdir("data/Training_crop/"+scene):
                # 1. load image
                curr_img = Image.open("data/Training_crop/"+scene+"/"+image)
                data = np.array(curr_img)
                # 2. get a sequence of 6
                shape = data.shape
                new_data = data.reshape(1, shape[0], shape[1], shape[2])
                # print(sequence.shape)
                if sequence.size == 0:
                    sequence = new_data
                else:
                    sequence = np.append(sequence, new_data, axis=0)
                    seq_shape = sequence.shape
                    if seq_shape[0] == 6:
                        sequence = sequence.reshape(1, seq_shape[0], seq_shape[1], seq_shape[2], seq_shape[3])
                        ims = np.append(ims, sequence, axis=0) if ims.size > 0 else sequence
                        sequence = np.array([])
    return ims
