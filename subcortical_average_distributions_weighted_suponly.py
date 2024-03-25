# Nicola Dinsdale 2023
# Get embeddings and fit GMM to the embeddings
########################################################################################################################
# Import dependencies
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle
import torch.optim as optim
from utils import Args
import sys
import json
from torch.autograd import Variable
import random
import argparse


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

sites = [  'NYU', 'Leuven', 'Pitt', 'SDSU', 'Trinity', 'Yale' ]
weights = {'NYU':129, 'Leuven':37, 'Pitt':39, 'SDSU':24, 'Trinity':33, 'Yale':33}
total = 129+37+39+24+33+33

mu_store = np.zeros((5000, 1))
var_store  = np.zeros((5000, 1))
pi_store  = np.zeros((5000, 1))

for s in sites:
    MU_PATH = 'federated_model_' + s + '_iteration_' + str(iteration-1) + '_mu.npy'
    VAR_PATH = 'federated_model_' + s + '_iteration_' + str(iteration-1) + '_var.npy'
    PI_PATH = 'federated_model_' + s + '_iteration_' + str(iteration-1) + '_pi.npy'
    mu = np.load(MU_PATH)
    var = np.load(VAR_PATH)
    pi = np.load(PI_PATH)
    
    mu_store +=  (weights[s] * mu) / total
    var_store +=  (weights[s] * var) / total
    pi_store +=  (weights[s] * pi) / total

print(var_store)
print(var_store.shape)
print(pi_store.shape)

np.save('federated_model_aggregated_iteration_' + str(iteration-1) + '_mu.npy', mu)
np.save('federated_model_aggregated_iteration_' + str(iteration-1) + '_var.npy', var)
np.save('federated_model_aggregated_iteration_' + str(iteration-1) + '_pi.npy', pi)