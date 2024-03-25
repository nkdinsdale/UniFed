# Nicola Dinsdale 2023
# Get embeddings and fit GMM to the embeddings
########################################################################################################################
# Import dependencies
import numpy as np
from models.UNet_model import UNet3D, segmenter3D
from datasets.numpy_dataset import numpy_dataset
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
import random
import argparse
from GMM.gmm_EM import GaussianMixture

########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 300
args.batch_size = 1
args.alpha = 100
args.patience = 25
args.train_val_prop = 0.95
args.learning_rate = 1e-4

cuda = torch.cuda.is_available()
########################################################################################################################
parser = argparse.ArgumentParser(description='Define Inputs for harmonisation model')
parser.add_argument('-i', action="store", dest="Iteration")

results = parser.parse_args()
try:
    iteration = int(results.Iteration)
    print('Iteration : ', iteration)
except:
    raise Exception('Arguement not supplied')
########################################################################################################################
def get_emb(args, models, test_loader):
    cuda = torch.cuda.is_available()

    embeddings = []
    [encoder, regressor] = models
    encoder.eval()
    regressor.eval()

    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if list(data.size())[0] == args.batch_size:

                features = encoder(data)
                embeddings.append(features.detach().cpu().numpy())

    embeddings = np.array(embeddings)
    embeddings_cp = np.copy(embeddings)

    return embeddings_cp

def get_params(feat):
    feat = feat.reshape(-1, 5000, 1)
    #feat = feat.reshape(-1, 1)
    feat = torch.from_numpy(feat).cuda()
    model = GaussianMixture(5000, 1, 1, covariance_type="diag", init_params="random")
    model.fit(feat)
    pi = model.pi.cpu().detach().numpy().squeeze().reshape(5000, 1)
    var = model.var.cpu().detach().numpy().squeeze().reshape(5000, 1)
    mu = model.mu.cpu().detach().numpy().squeeze().reshape(5000, 1)
    return pi, var, mu

########################################################################################################################
if iteration == 1:
    LOAD_PATH_UNET = 'NYU_unet_checkpoint'
    LOAD_PATH_SEGMENTER = 'NYU_segmenter_checkpoint'
else:
    LOAD_PATH_UNET = 'unet_aggreated_' + str(iteration-1)
    LOAD_PATH_SEGMENTER =  'segmenter_aggreated_' + str(iteration-1)

sites = ['UCLA', 'Yale',  'Trinity', 'Stanford', 'SDSU', 'Pitt',  'NYU', 'MaxMun', 'Leuven', 'KKI', 'Caltech']

for s in sites: 
    MU_SAVE_PATH = 'federated_model_' + s + '_iteration_' + str(iteration-1) + '_mu'
    VAR_SAVE_PATH = 'federated_model_' + s + '_iteration_' + str(iteration-1) + '_var'
    PI_SAVE_PATH = 'federated_model_' + s + '_iteration_' + str(iteration-1) + '_pi'

    X = np.load('X_' + s + '_train.npy')
    y = np.load('y_' + s + '_train.npy').astype(int)

    X, y = np.reshape(X, (-1, 1, 128, 240, 160)), np.reshape(y, (-1, 128, 240, 160))

    y_store = np.zeros((5, y.shape[0], 128, 240, 160))
    print(y.shape)
    print(y_store.shape)
    print(np.unique(y))
    y_store[0,:,:,:,:][y==0] = 1
    y_store[0,:,:,:,:][y==11] = 1
    y_store[0,:,:,:,:][y==50] = 1
    y_store[0,:,:,:,:][y==13] = 1
    y_store[0,:,:,:,:][y==52] = 1
    y_store[0,:,:,:,:][y==26] = 1
    y_store[0,:,:,:,:][y==58] = 1
    y_store[0,:,:,:,:][y==18] = 1
    y_store[0,:,:,:,:][y==54] = 1

    y_store[1,:,:,:,:][y==10] = 1
    y_store[1,:,:,:,:][y==49] = 1
    y_store[2,:,:,:,:][y==12] = 1
    y_store[2,:,:,:,:][y==51] = 1
    y_store[3,:,:,:,:][y==17] = 1
    y_store[3,:,:,:,:][y==53] = 1
    y_store[4,:,:,:,:][y==16] = 1
    y = y_store
    y = np.transpose(y, (1, 0, 2, 3, 4))
    print(y.shape)

    print(X.shape)
    print('Testing: ', X.shape, y.shape, flush=True)
        
    print('Creating datasets and dataloaders')
    test_dataset = numpy_dataset(X, y)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load the model
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
        
    models = [unet, segmenter]
    print('EMBEDDINGS')
    embeddings = get_emb(args, models, test_dataloader)
    embeddings = np.reshape(embeddings, (embeddings.shape[0], -1))

    top_indexs = np.load('top_indexs.npy')
    embeddings = embeddings[:, top_indexs]
    
    pi, var, mu = get_params(embeddings)
    print(pi.shape)
    print(var.shape)
    print(mu.shape)
    
    np.save(PI_SAVE_PATH, pi)
    np.save(VAR_SAVE_PATH, var)
    np.save(MU_SAVE_PATH, mu)
