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
train_dl = utils.make_dataloaders(paths=train_paths, split='train')
val_dl = utils.make_dataloaders(paths=val_paths, split='val')

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']


# discriminator = base_model.PatchDiscriminator(3) 
# out = discriminator(params.dummy_input)

gan_model = model.MainModel(attention=params.use_cbam)
print(gan_model)

model.train(gan_model, train_dl, val_dl, 600)



torch.save(model.state_dict(), params.model_path+"/"+params.model_name)
