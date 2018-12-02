import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helper
import simulation
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image, ImageOps
import glob
import cv2
from extract_patches import get_data_training
from help_functions import *
import math
from loss import dice_loss
from calculate_gc import cal_gc as cal_gc
from tqdm import tqdm as tqdm  





## Generate some random images
#input_images, target_masks = simulation.generate_random_data(192, 192, count=3)
#
#for x in [input_images, target_masks]:
#    print(x.shape)
#    print(x.min(), x.max())
#
## Change channel-order and make 3 channels for matplot
#input_images_rgb = [x.astype(np.uint8) for x in input_images]
#
## Map each channel (i.e. class) to each color
#target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
#
## Left: Input image, Right: Target mask (Ground-truth)
#helper.plot_side_by_side([input_images_rgb, target_masks_rgb])
#

#===========Prepare Dataset and DataLoader===========
#format (Number, Channel, Weight, height)

class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.transform = transform


        path_data = './DRIVE_datasets_training_testing/'
        #============ Load the data and divided in patches
        patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original = path_data + ('DRIVE_dataset_imgs_train.hdf5'),
        DRIVE_train_groudTruth = path_data + ('DRIVE_dataset_groundTruth_train.hdf5'),  #masks
        patch_height = 48,
        patch_width = 48,
        N_subimgs = 190000,
        inside_FOV = False #select the patches only inside the FOV  (default == True)
        )
        
        patches_imgs_train = np.transpose(patches_imgs_train,(0,2,3,1))
#        patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption

        self.input_images=patches_imgs_train
        self.target_masks=patches_masks_train
    
        #========= Save a sample of what you're feeding to the neural network ==========
#        N_sample = min(patches_imgs_train.shape[0],40)
#        visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./images/'+"sample_input_imgs")#.show()
#        visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./images/'+"sample_input_masks")#.show()
        #visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],1),'./'+name_experiment+'/'+"sample_input_imgs").show()

    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)
        
        return [image, mask]

# use same transform for train/val for this example
trans = transforms.Compose([
    transforms.ToTensor(),
])

train_set = SimDataset(2000, transform = trans)
#val_set = SimDataset(200, transform = trans)

val_set = train_set
print(len(train_set))
print(len(val_set))

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 32

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

dataset_sizes

#====================Check the outputs from DataLoader========================
import torchvision.utils

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    
    return inp

# Get a batch of training data
inputs, masks = next(iter(dataloaders['train']))

#print(inputs.shape, masks.shape)
#for x in [inputs.numpy(), masks.numpy()]:
#    print(x.min(), x.max(), x.mean(), x.std())
#
#plt.imshow(reverse_transform(inputs[3]))

#========================Create the UNet module========================================--
from torchsummary import summary
import torch
import torch.nn as nn
import pytorch_unet_gray

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = pytorch_unet_gray.UNet(1)
model = model.to(device)
#model = model.float().to(device)
summary(model, input_size=(1, 48, 48))

#============================Define the main training loop========================================
from collections import defaultdict
import torch.nn.functional as F

def calc_loss(pred, target, metrics, bce_weight=0.5):
    
    pred=pred.type('torch.FloatTensor')
    target=target.type('torch.FloatTensor')

    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    gc_loss  = cal_gc(pred,target)
    loss = bce * bce_weight + dice * (1 - bce_weight)+gc_loss
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['gc_loss'] += gc_loss.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs=inputs.type('torch.cuda.FloatTensor')

                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#====================================Training===================================
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_class = 1

model = pytorch_unet_gray.UNet(num_class).to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=50)
torch.save(model, "model.pt")
#=============================================================================
#=============================================================================
#=============================================================================
#==============================================================================!
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from pre_processing import my_PreProc
# prediction

path_data = './DRIVE_datasets_training_testing/'

#original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + ('DRIVE_dataset_imgs_test.hdf5')
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]


#the border masks provided by the DRIVE
DRIVE_test_border_masks = path_data + ('DRIVE_dataset_borderMasks_test.hdf5')
test_border_masks = load_hdf5(DRIVE_test_border_masks)

# dimension of the patches
patch_height = 48
patch_width = 48

#the stride in case output with average
stride_height = 5
stride_width = 5
assert (stride_height < patch_height and stride_width < patch_width)

Imgs_to_test = 20
#Grouping of the predicted images
N_visual = 1
#====== average mode ===========
average_mode = True

#============ Load the data and divide in patches

patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
    DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
    DRIVE_test_groudTruth = path_data + ('DRIVE_dataset_groundTruth_test.hdf5'),
    Imgs_to_test = Imgs_to_test, patch_height = patch_height,
    patch_width = patch_width,
    stride_height = stride_height,
    stride_width = stride_width
)
 
model.eval() # Set model to evaluate mode

for xx in [patches_imgs_test, masks_test]:
    print(xx.shape)
    print(xx.min(), xx.max())

test_dataset =[xx for xx in patches_imgs_test]

test_label = [masks_test]

#test_dataset = SimDataset(3, transform = trans, tipe="test")
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)
 
inputs, labels = next(iter(test_loader))
#inputs= next(iter(test_loader))
inputs = inputs.to(device, dtype=torch.float)
labels = labels.to(device)
print(inputs.shape)
pred = model(inputs)

pred = pred.data.cpu().numpy()
print(pred.shape)
np.save("hasil", pred)

#===== Convert the prediction arrays in corresponding images
#pred_patches = pred_to_imgs(pred, patch_height, patch_width, "original")
pred_patches = pred

#========== Elaborate and visualize the predicted images ====================
pred_imgs = None
orig_imgs = None
gtruth_masks = None
pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])    #originals
gtruth_masks = masks_test  #ground truth masks


