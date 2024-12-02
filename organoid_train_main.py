import torch
import numpy as np
from utils import utils, params
from model import model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


img_count = len(params.paths)
train_samples = (img_count//params.train_per)*params.train_per
print('train samples: ', train_samples)
np.random.seed(123)
paths_subset = np.random.choice(params.paths, img_count, replace=False) # choosing 10000 images randomly
rand_idxs = np.random.permutation(img_count)
train_idxs = rand_idxs[:-1] # choosing the first x samples as training set
val_idxs = rand_idxs[-1:] # choosing last y samples as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print("# of Train paths", len(train_paths), "# of Val paths", len(val_paths))

# loading dataset
if params.gen_individual_fl:
    train_dl = utils.custom_dataloaders(paths=train_paths, split='train')
else:
    train_dl = utils.make_dataloaders(paths=train_paths, split='train')

val_dl = utils.make_dataloaders(paths=val_paths, split='val')

gan_model = model.MainModel(attention=params.use_cbam)
print(gan_model)

model.train(gan_model, train_dl, val_dl, epochs=params.epochs ,generator_steps=params.generator_steps, discriminator_steps=params.discriminator_steps)


torch.save(gan_model.state_dict(), params.model_path+"/"+params.model_name)