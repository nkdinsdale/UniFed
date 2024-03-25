# Nicola Dinsdale 2023
# Main file for subcortical segmentation model on a subset of the data
########################################################################################################################
# Import dependencies
import numpy as np
from models.UNet_model import UNet3D, segmenter3D
from datasets.numpy_dataset import numpy_dataset_three
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from losses.dice_loss import dice_loss
from sklearn.utils import shuffle
import torch.optim as optim
from utils import Args, EarlyStopping_split_models, load_pretrained_model
import sys
import json
from torch.autograd import Variable
import argparse
from losses.dice_cce_loss import DiceAndCE
from losses.bhattacharya import bhattachayra_GMM
from GMM.gmm_EM import GaussianMixture

########################################################################################################################
parser = argparse.ArgumentParser(description='Define Inputs for harmonisation model')
parser.add_argument('-s', action="store", dest="Site")
parser.add_argument('-i', action="store", dest="Iteration")


results = parser.parse_args()
try:
    site = str(results.Site)
    print('Training Site : ', site)
    iteration = int(results.Iteration)
    print('Current Iteration : ', iteration)

except:
    raise Exception('Arguement not supplied')
########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 1000
args.batch_size = 5
args.alpha = 100
args.patience = 5
args.train_val_prop = 0.80
args.learning_rate = 1e-5

cuda = torch.cuda.is_available()

if iteration == 1:
    LOAD_PATH_UNET = 'NYU_unet_checkpoint'
    LOAD_PATH_SEGMENTER = 'NYU_segmenter_checkpoint'
else:
    LOAD_PATH_UNET = 'unet_aggreated_' + str(iteration-1)
    LOAD_PATH_SEGMENTER =  'segmenter_aggreated_' + str(iteration-1)

CHK_PATH_UNET = site + '_unet_checkpoint_iteration_' + str(iteration)
CHK_PATH_SEGMENTER = site + '_segmenter_checkpoint_iteration_' + str(iteration)
LOSS_PATH = site + '_only_losses_iteration_'  + str(iteration)

ref_mu = np.load('federated_model_aggregated_iteration_' + str(iteration-1) + '_mu.npy')
ref_sigma = np.sqrt(np.load('federated_model_aggregated_iteration_' + str(iteration-1) + '_var.npy'))
ref_pi = np.load('federated_model_aggregated_iteration_' + str(iteration-1) + '_pi.npy')

print('Reference Distributions: ')
print('Mu: ', ref_mu.shape)
print('Sigma: ', ref_sigma.shape)
print('Pi: ', ref_pi.shape)

args.indexs = np.load('top_indexs.npy')
print(args.indexs.shape)

########################################################################################################################
def train_normal(args, models, train_loader, optimizer, criterions, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor] = models
    [dice_criterion, bhatt_criterion] = criterions
    total_loss = 0
    gmm_model = GaussianMixture(5000, 1, 1, covariance_type="diag", init_params="random")

    encoder.train()
    regressor.train()

    batches = 0
    for batch_idx, (data, target, un_data) in enumerate(train_loader):
        if cuda:
            data, target, un_data = data.cuda(), target.cuda(), un_data.cuda()
        data, target, un_data = Variable(data), Variable(target), Variable(un_data)

        batches += 1

        # First update the encoder and regressor
        optimizer.zero_grad()
        features = encoder(data)
        shape_store = features.shape

        features = features.view(args.batch_size, -1, 1)
        features_reduced = features[:, args.indexs, :]
        gmm_model.fit(features_reduced, n_iter=20)
        b_loss = bhatt_criterion(gmm_model.mu, torch.sqrt(gmm_model.var), gmm_model.pi)
        
        pred = regressor(features.view(shape_store))        
        d_loss = dice_criterion(target, pred)
        
        loss = d_loss + 0.01 * b_loss 
            
        loss.backward()
        optimizer.step()

        total_loss += loss

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                        100. * (batch_idx+1) / len(train_loader), loss.item()), flush=True)
        del loss
        del features

    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())

    del av_loss

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss_copy,  flush=True))

    return av_loss_copy, np.NaN

def val_normal(args, models, val_loader, criterions):

    cuda = torch.cuda.is_available()

    [encoder, regressor] = models
    [dice_criterion, bhatt_criterion] = criterions

    gmm_model = GaussianMixture(5000, 1, 1, covariance_type="diag", init_params="random")

    encoder.eval()
    regressor.eval()

    total_loss = 0

    batches = 0
    with torch.no_grad():

        for batch_idx, (data, target, un_data) in enumerate(val_loader):
            if cuda:
                data, target, un_data = data.cuda(), target.cuda(), un_data.cuda()
            data, target, un_data = Variable(data), Variable(target), Variable(un_data)

            batches += 1
            features = encoder(data)
            shape_store = features.shape

            features = features.view(args.batch_size, -1, 1)
            features_reduced = features[:, args.indexs, :]
            gmm_model.fit(features_reduced, n_iter=20)
            b_loss = bhatt_criterion(gmm_model.mu, torch.sqrt(gmm_model.var), gmm_model.pi)
            
            pred = regressor(features.view(shape_store))        
            d_loss = dice_criterion(target, pred)
            print(b_loss)
            print(d_loss)
            loss = d_loss + 0.01 * b_loss 

            total_loss  += loss

    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())
    del av_loss

    print('Validation set: Average loss: {:.4f}\n'.format(av_loss_copy,  flush=True))

    return av_loss_copy, np.NaN
########################################################################################################################

X = np.load('X_' + site + '_train.npy')
y = np.load('y_' + site + '_train.npy')
X, y = shuffle(X, y, random_state=0)
print('Orig Shape = ', X.shape)

proportion = int(args.train_val_prop * len(X))
X_train, y_train = X[:proportion], y[:proportion]
n_subjs = X_train.shape[0]

X_val, y_val = X[proportion:], y[proportion:]

print('Training: ', X_train.shape, y_train.shape, flush=True)
print('Validation: ', X_val.shape, y_val.shape, flush=True)

print('Keeping subset of labelled data')
proportion = int(0.01 * len(X))
X_train = X_train[:proportion+1]
y_train = y_train[:proportion+1]

X_train, y_train = np.reshape(X_train, (-1, 1, 128, 240, 160)), np.reshape(y_train, (-1, 128, 240, 160))
X_val, y_val = np.reshape(X_val, (-1, 1, 128, 240, 160)), np.reshape(y_val, (-1, 128, 240, 160))

y_store = np.zeros((5, y_train.shape[0], 128, 240, 160))
print(y_train.shape)
print(y_store.shape)
print(np.unique(y_train))
y_store[0,:,:,:,:][y_train==0] = 1
y_store[0,:,:,:,:][y_train==11] = 1
y_store[0,:,:,:,:][y_train==50] = 1
y_store[0,:,:,:,:][y_train==13] = 1
y_store[0,:,:,:,:][y_train==52] = 1
y_store[0,:,:,:,:][y_train==26] = 1
y_store[0,:,:,:,:][y_train==58] = 1
y_store[0,:,:,:,:][y_train==18] = 1
y_store[0,:,:,:,:][y_train==54] = 1

y_store[1,:,:,:,:][y_train==10] = 1
y_store[1,:,:,:,:][y_train==49] = 1
y_store[2,:,:,:,:][y_train==12] = 1
y_store[2,:,:,:,:][y_train==51] = 1
y_store[3,:,:,:,:][y_train==17] = 1
y_store[3,:,:,:,:][y_train==53] = 1
y_store[4,:,:,:,:][y_train==16] = 1
y_train = y_store
y_train = np.transpose(y_train, (1, 0, 2, 3, 4))
print(y_train.shape)

y_store = np.zeros((5, y_val.shape[0], 128, 240, 160))
y_store[0,:,:,:,:][y_val==0] = 1
y_store[0,:,:,:,:][y_val==11] = 1
y_store[0,:,:,:,:][y_val==50] = 1
y_store[0,:,:,:,:][y_val==13] = 1
y_store[0,:,:,:,:][y_val==52] = 1
y_store[0,:,:,:,:][y_val==26] = 1
y_store[0,:,:,:,:][y_val==58] = 1
y_store[0,:,:,:,:][y_val==16] = 1
y_store[0,:,:,:,:][y_val==18] = 1
y_store[0,:,:,:,:][y_val==54] = 1

y_store[1,:,:,:,:][y_val==10] = 1
y_store[1,:,:,:,:][y_val==49] = 1
y_store[2,:,:,:,:][y_val==12] = 1
y_store[2,:,:,:,:][y_val==51] = 1
y_store[3,:,:,:,:][y_val==17] = 1
y_store[3,:,:,:,:][y_val==53] = 1
y_store[4,:,:,:,:][y_val==16] = 1
y_val = y_store
y_val = np.transpose(y_val, (1, 0, 2, 3, 4))

print('Training: ', X_train.shape, y_train.shape, flush=True)
print('Validation: ', X_val.shape, y_val.shape, flush=True)

# Upsample training dataset to make full size
print('Upsampling dataset')
X_train_orig = np.copy(X_train)
y_train_orig = np.copy(y_train)
while X_train.shape[0] < n_subjs:
    X_train = np.append(X_train, X_train_orig, axis=0)
    y_train = np.append(y_train, y_train_orig, axis=0)
    print('Training: ', X_train.shape, y_train.shape, flush=True)

print('Creating unlearning dataset')
X_un = np.load('/home/bras3596/data/ABIDE/subcortical/X_' + site + '_train.npy')

print('Orig Shape = ', X_un.shape)

X_un = shuffle(X_un, random_state=0)

proportion = int(args.train_val_prop * len(X_un))
X_un_train = X_un[:proportion]
X_un_val = X_un[proportion:]

print('Training: ', X_un_train.shape, flush=True)
print('Validation: ', X_un_val.shape, flush=True)

X_un_train = np.reshape(X_un_train, (-1, 1, 128, 240, 160))
X_un_val = np.reshape(X_un_val, (-1, 1, 128, 240, 160))

print('Training: ', X_un_train.shape, flush=True)
print('Validation: ', X_un_val.shape, flush=True)

X_un_shape = X_un_train.shape[0]
X_shape = X_train.shape[0]
if X_un_shape > X_shape:
    X_un_train = X_un_train[:X_shape]
elif X_shape > X_un_shape:
    X_train = X_train[:X_un_shape]
    y_train = y_train[:X_un_shape]

print('Training: ', X_train.shape, y_train.shape,  X_un_train.shape, flush=True)

print('Creating datasets and dataloaders')
train_dataset = numpy_dataset_three(X_train, y_train, X_un_train)
val_dataset = numpy_dataset_three(X_val, y_val, X_un_val)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

# Load the model
print('\nCreating the models')
unet = UNet3D(init_features=8)
segmenter = segmenter3D(out_channels=5, init_features=8)

if cuda:
    unet = unet.cuda()
    segmenter = segmenter.cuda()

if LOAD_PATH_UNET:
   print('Loading Weights')
   encoder_dict = unet.state_dict()
   pretrained_dict = torch.load(LOAD_PATH_UNET)
   pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
   print('weights loaded unet = ', len(pretrained_dict), '/', len(encoder_dict))
   unet.load_state_dict(torch.load(LOAD_PATH_UNET))
if LOAD_PATH_SEGMENTER:
   print('Loading Weights')
   encoder_dict = segmenter.state_dict()
   pretrained_dict = torch.load(LOAD_PATH_SEGMENTER)
   pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
   print('weights loaded segmenter = ', len(pretrained_dict), '/', len(encoder_dict))
   segmenter.load_state_dict(torch.load(LOAD_PATH_SEGMENTER))

bhatt_criterion = bhattachayra_GMM(1, ref_mu, ref_sigma, ref_pi)
dice_criterion = dice_loss()
if cuda:
    bhatt_criterion = bhatt_criterion.cuda()
    dice_criterion = dice_criterion.cuda()

optimizer = optim.AdamW(list(unet.parameters()) + list(segmenter.parameters()), lr=args.learning_rate)
# Initalise the early stopping
early_stopping = EarlyStopping_split_models(args.patience, verbose=False)

epoch_reached = 1
loss_store = []

models = [unet, segmenter]
criterions = [dice_criterion, bhatt_criterion]

for epoch in range(epoch_reached, args.epochs+1):
    print('Epoch ', epoch, '/', args.epochs, flush=True)
    loss, _ = train_normal(args, models, train_dataloader, optimizer, criterions, epoch)

    val_loss, _ = val_normal(args, models, val_dataloader, criterions)
    loss_store.append([loss, val_loss])
    np.save(LOSS_PATH, np.array(loss_store))

    # Decide whether the model should stop training or not
    early_stopping(val_loss, models, epoch, optimizer, loss, [CHK_PATH_UNET, CHK_PATH_SEGMENTER])

    if early_stopping.early_stop:
        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)
        sys.exit('Patience Reached - Early Stopping Activated')

    if epoch == args.epochs:
        print('Finished Training', flush=True)
        print('Saving the model', flush=True)

        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)

    torch.cuda.empty_cache()  # Clear memory cache