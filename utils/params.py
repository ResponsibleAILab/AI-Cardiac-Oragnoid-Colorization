import os
import glob
import torch

SIZE = 256
batch_size = 16
dummy_input = torch.randn(batch_size, 3, 256, 256) # batch_size, channels, size, size

# Models
# CBAM params
use_cbam = True    # If false then its only basic Pix2Pix architecture
epochs = 600
# Generator Iterations params - when both are 1, paramaters will get updated once in the corresponding network 
generator_steps = 1     # change this to 2 for generator iterations model (making generatro stronger compared to discriminator)
discriminator_steps = 1
gen_individual_fl = True # False if GAN should generate fluorescence of all 3 channels. True if GAN should generate fluorescence of one channel

# Dataset
dir = os.getcwd()
main_path = '/dataset/allDataset_main_excluded_darkImgs/overlay/'
fluorescence_path = '/dataset/allDataset_main_excluded_darkImgs/green/'
path = os.path.join(dir, main_path)

#saving the model
model_name = "cbam_green_cahannel.pt"
model_path = os.path.join(dir, "saved_models")
train_per = 80 # percent of training images from the dataset

paths1 = glob.glob(path+'*.png') # Grabbing all the image types in the directory
paths2 = glob.glob(path+'*.jpg')
paths3 = glob.glob(path+'*.tif')

paths = paths1+paths2+paths3