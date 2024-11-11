import os
import glob
import torch

SIZE = 256
dummy_input = torch.randn(16, 3, 256, 256) # batch_size, channels, size, size

# Models
# CBAM params
use_cbam = True    # If false then its only basic Pix2Pix architecture
# Generator Iterations params - when both are 1, paramaters will get updated once in the corresponding network 
generator_steps = 1     # change this to 2 for generator iterations model (making generatro stronger compared to discriminator)
discriminator_steps = 1

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

