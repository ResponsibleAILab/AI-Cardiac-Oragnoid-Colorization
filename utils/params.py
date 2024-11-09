import os
import glob
import torch

SIZE = 256
dummy_input = torch.randn(16, 3, 256, 256) # batch_size, channels, size, size

use_cbam = True

# Dataset
dir = os.getcwd() 
path = os.path.join(dir, '/dataset/allDataset_main_excluded_darkImgs/')

#saving the model
model_name = "cbam.pt"
model_path = os.path.join(dir, "/models")
train_per = 80 # percent of training images from the dataset

paths1 = glob.glob(path+'*.png') # Grabbing all the image types in the directory
paths2 = glob.glob(path+'*.jpg')
paths3 = glob.glob(path+'*.tif')

paths = paths1+paths2+paths3

