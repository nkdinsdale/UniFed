# Nicola Dinsdale 2021
# Federated Average Script
########################################################################################################################
# Import dependencies
import numpy as np
from models.UNet_model import UNet3D, segmenter3D
import argparse
import torch
from collections import OrderedDict

cuda = torch.cuda.is_available()
########################################################################################################################
parser = argparse.ArgumentParser(description='Define Inputs for pruning model')
parser.add_argument('-i', action="store", dest="Iteration")
results = parser.parse_args()
try:
    iteration = int(results.Iteration)
    print('Current Iteration : ', iteration)
except:
    raise Exception('Arguement not supplied')

PATH_UNET = 'unet_aggreated_' + str(iteration)
PATH_SEGMENTER = 'segmenter_aggreated_' + str(iteration)

########################################################################################################################
weights = {'UCLA': 1, 'Yale':1, 'Trinity':1,'Stanford':1, 'SDSU':1, 'Pitt':1, 'NYU':1, 'MaxMun':1, 'Leuven':1, 'KKI':1, 'Caltech':1}
total = 11
sites = ['UCLA', 'Yale',  'Trinity', 'Stanford', 'SDSU', 'Pitt',  'NYU', 'MaxMun', 'Leuven', 'KKI', 'Caltech']

unet_pths = []
segmenter_pths = []
for i, site in enumerate(sites):
    unet_pths.append( site + '_unet_checkpoint_iteration_' + str(iteration))
    segmenter_pths.append(site + '_segmenter_checkpoint_iteration_'  + str(iteration))
    
unet_pths = np.array(unet_pths)
segmenter_pths = np.array(segmenter_pths)

update_state_unet = OrderedDict()
update_state_segmenter = OrderedDict()

unet = UNet3D(init_features=8)
segmenter = segmenter3D(out_channels=5, init_features=8)

if cuda:
    unet = unet.cuda()
    segnet = segmenter.cuda()

for i, site in enumerate(sites):
    unet.load_state_dict(torch.load(unet_pths[i]))
    segnet.load_state_dict(torch.load(segmenter_pths[i]))

    local_state_unet = unet.state_dict()
    local_state_segnet = segnet.state_dict()
    for key in unet.state_dict().keys():
        if i == 0:
            update_state_unet[key] = local_state_unet[key] * (weights[site]/total)
        else:
            update_state_unet[key] += local_state_unet[key] * (weights[site] / total)
    for key in segnet.state_dict().keys():
        if i == 0:
            update_state_segmenter[key] = local_state_segnet[key] * (weights[site]/total)
        else:
            update_state_segmenter[key] += local_state_segnet[key] * (weights[site]/total)

unet.load_state_dict(update_state_unet)
segnet.load_state_dict(update_state_segmenter)

torch.save(unet.state_dict(), PATH_UNET)
torch.save(segnet.state_dict(), PATH_SEGMENTER)


