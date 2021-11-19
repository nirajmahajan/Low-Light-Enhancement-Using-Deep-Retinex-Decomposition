import numpy as np
import torch
import random
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
from torch import nn, optim
from torch.nn import functional as F
import pickle
import argparse
import sys
import os
os.environ['display'] = 'localhost:14.0'

import PIL
import time
from tqdm import tqdm
import gc

from myModels import LowLightEnhancer
from myDatasets import LOLDataset

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--cuda', type = int, default = '-1')
args = parser.parse_args()

seed_torch(args.seed)

if args.cuda == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 20
lr = 1e-3
STEP_SIZE = 25
SAVE_INTERVAL = 1
NUM_EPOCHS = 100
torch.autograd.set_detect_anomaly(True)

checkpoint_path = './checkpoints/'
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
if not os.path.isdir('../results/'):
    os.mkdir('../results/')
if not os.path.isdir('../results/known'):
    os.mkdir('../results/known')
if not os.path.isdir('../results/unknown'):
    os.mkdir('../results/unknown')

# train_transform = transforms.Compose([transforms.Resize((224,224))])
test_transform = transforms.Compose([transforms.Resize((224,224))])
train_transform = None
# test_transform = None

trainset = LOLDataset(train = 'train', transform=train_transform, p_rot90 = 0.5, p_flipud = 0.5, p_fliplr = 0.5, patch_mode = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle = True)
testset = LOLDataset(train = 'test', transform=test_transform, p_rot90 = 0, p_flipud = 0, p_fliplr = 0, patch_mode = False)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle = False)
unknonwset = LOLDataset(train = 'unknown', transform=test_transform, p_rot90 = 0, p_flipud = 0, p_fliplr = 0, patch_mode = False)
unknownloader = torch.utils.data.DataLoader(unknonwset, batch_size=TEST_BATCH_SIZE, shuffle = False)
model = LowLightEnhancer(optim_choice = 'Adam', lr = lr, device = device).to(device)

if args.resume:
    model_state = torch.load(checkpoint_path + 'state.pth', map_location = device)['state']
    if (not args.state == -1):
        model_state = args.state
    print('Loading checkpoint at model state {}'.format(model_state))
    dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state), map_location = device)
    pre_e = dic['e']
    model.load_state_dict(dic['model'])
    # model.relight_optimizer.load_state_dict(dic['relight_optimizer'])
    # model.decom_optimizer.load_state_dict(dic['decom_optimizer'])
    model.optimizer.load_state_dict(dic['optimizer'])
    decom_losses = dic['decom_losses']
    relight_losses = dic['relight_losses']
    print('Resuming Training after {} epochs'.format(pre_e))
else:
    model_state = 0
    pre_e =0
    decom_losses = []
    relight_losses = []
    print('Starting Training')

def train(e):
    print('\nTraining for epoch {}'.format(e))
    tot_loss_decom = 0
    tot_loss_relight = 0

    for batch_num,(S_low, S_high) in tqdm(enumerate(trainloader), desc = 'Epoch {}'.format(e), total = len(trainloader)):
        L_Decom, L_Relight = model.train(S_low, S_high)
        tot_loss_decom += L_Decom
        tot_loss_relight += L_Relight

    print('Total Decomposition Loss for epoch = {}'.format(tot_loss_decom/batch_num))
    print('Total Relighting Loss for epoch = {}'.format(tot_loss_relight/batch_num), flush = True)
    return tot_loss_decom/batch_num, tot_loss_relight/batch_num

def evaluate(e, known = "True"):
    if known:
        str_k = 'known'
        dloader = testloader
    else:
        str_k = 'unknown'
        dloader = unknownloader
    total = 0
    with torch.no_grad():
        for batch_num,(S_low, S_high) in tqdm(enumerate(dloader), desc = 'Testing on {} images'.format(str_k), total = len(dloader)):
            S_corrected, Ihat, Rhat = model.evaluate(S_low)
            for i in range(S_low.shape[0]):
                im_low = S_low[i,:,:,:].permute(1,2,0)
                im_high = S_high[i,:,:,:].permute(1,2,0)
                im_corrected = S_corrected[i,:,:,:]
                im_ill = Ihat[i,:,:,0]
                im_ref = Rhat[i,:,:,:]
                # im_low -= im_low.min()
                # im_low /= im_low.max()
                im_high -= im_high.min()
                im_high /= im_high.max()
                im_corrected -= im_corrected.min()
                im_corrected /= im_corrected.max()
                im_ill -= im_ill.min()
                im_ill /= im_ill.max()
                im_ref -= im_ref.min()
                im_ref /= im_ref.max()

                plt.imsave("../results/R.png", im_ref)
                plt.imsave("../results/I.png", im_ill, cmap = 'gray')
                
                fig = plt.figure(figsize = (15,3))
                plt.subplot(1,5,1)
                plt.imshow(im_low)
                plt.title('Low Light Image')
                plt.axis('off')
                plt.subplot(1,5,2)
                plt.imshow(im_corrected)
                plt.title('Corrected Image')
                plt.axis('off')
                plt.subplot(1,5,3)
                plt.imshow(im_high)
                plt.title('High Light Image')
                plt.axis('off')
                plt.subplot(1,5,4)
                plt.imshow(im_ill, cmap = 'gray')
                plt.title('Illuminance')
                plt.axis('off')
                plt.subplot(1,5,5)
                plt.imshow(im_ref)
                plt.title('Reflectance')
                plt.axis('off')
                plt.suptitle("After {} epochs".format(e))
                plt.savefig('../results/{}/{}.png'.format(str_k, i+total))
                plt.close('all')
            total += S_corrected.shape[0]

if args.eval:
    evaluate(pre_e, known = False)
    evaluate(pre_e, known = True)
    with open('status.txt', 'w') as f:
        f.write('1')
    os._exit(0)

for e in range(NUM_EPOCHS):
    # model_state = e//50
    if pre_e > 0:
        pre_e -= 1
        continue

    # if e % SAVE_INTERVAL == 0:
    #     seed_torch(args.seed)

    l_decom, l_relight = train(e)
    decom_losses.append(l_decom)
    relight_losses.append(l_relight)
    
    dic = {}
    dic['e'] = e+1
    dic['model'] = model.state_dict()
    # dic['decom_optimizer'] = model.decom_optimizer.state_dict()
    # dic['relight_optimizer'] = model.relight_optimizer.state_dict()
    dic['optimizer'] = model.optimizer.state_dict()
    dic['decom_losses'] = decom_losses
    dic['relight_losses'] = relight_losses


    if (e+1) % SAVE_INTERVAL == 0:
        torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
        torch.save({'state': model_state}, checkpoint_path + 'state.pth')
        print('Saving model after {} Epochs'.format(e+1))
        evaluate(e, known = False)
        evaluate(e, known = True)