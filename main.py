import time
import numpy as np

from libs import dataset_niftynet as dset_utils
from libs import loss as loss_utils
from libs import model as cnn_utils

from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.layer.mean_variance_normalisation import MeanVarNormalisationLayer
from niftynet.layer.rand_rotation import RandomRotationLayer as Rotate
from niftynet.engine.signal import TRAIN, VALID, INFER
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


def get_sampler(data_param,image_sets_partitioner,phase,patch_size):


    # Using Nifty Reader and Sampler to create a dataset for PyTorch DataLoader
    if phase == 'train':
        image_reader = ImageReader().initialise(data_param,
                                                file_list=image_sets_partitioner.get_file_list(TRAIN))

    elif phase == 'validation':
        image_reader = ImageReader().initialise(data_param,
                                                file_list=image_sets_partitioner.get_file_list(VALID))
    else:
        raise Exception('Invalid phase choice: {}'.format({'phase':['train','validation']}))


    mean_variance_norm_layer = MeanVarNormalisationLayer(image_name='image')
    rotation_layer = Rotate()
    rotation_layer.init_uniform_angle([-10.0, 10.0])

    image_reader.add_preprocessing_layers([mean_variance_norm_layer]) # preprocessing
    image_reader.add_preprocessing_layers([rotation_layer]) # augmentation


    sampler = UniformSampler(image_reader,
                             window_sizes=(patch_size, patch_size, patch_size),
                             windows_per_image=1)

    return sampler

def train(dataloaders, dataset_sizes,model,criterion,optimizer,num_epochs,device,in_channels):

    since = time.time()
    model = model.to(device)


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for iteration, (inputs, labels) in enumerate(dataloaders[phase], 1):

                nbatches, wsize, x, y, z = inputs.size()
                inputs = inputs.view(nbatches * wsize, in_channels, x, y, z)
                labels = labels.view(nbatches * wsize, in_channels, x, y, z)
                labels = (labels > 0.5).float()  # STAPLE outputs a prob. map

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)[1]
                    outputs = F.sigmoid(outputs)  # To compute Soft Dice loss
                    pred = (outputs > 0.5).float()

                    los = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        los.backward()
                        optimizer.step()

                # statistics
                running_loss += los.item()
                running_corrects += loss_utils.dice(pred, labels)

                print('Iteration: {} batch loss {:.4f}'.format(iteration, los.item()))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if epoch == 0:
                best_loss = epoch_loss
                torch.save(model.state_dict(), './CP{}.pth'.format(epoch + 1))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), './CP{}.pth'.format(epoch + 1))
                print('Checkpoint {} saved !'.format(epoch + 1))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def inference(data_param, image_sets_partitioner, patch_size, window_border, model, device):

    image_reader_test = ImageReader(names='image').initialise(data_param,
                                                              file_list=image_sets_partitioner.get_file_list(INFER))
    sampler_test = GridSampler(image_reader_test,
                               window_sizes=(patch_size, patch_size, patch_size),
                               window_border=window_border,
                               batch_size = 1)

    output_decoder = GridSamplesAggregator(image_reader=sampler_test.reader,
                                           output_path='/home/oeslle/Documents/pred')

    model.load_state_dict(torch.load('./CP2.pth'))
    model.to(device)
    model.eval()

    for batch_output in sampler_test():
        window =  batch_output['image'][...,0,:] # [...,0,:] eliminates time coordinate from NiftyNet Volume

        nb, x, y, z, nc = window.shape
        window = window.reshape(nb, nc, x, y, z)
        window = torch.Tensor(window).to(device)

        with torch.no_grad():
            outputs = model(window)[1]
            outputs = F.sigmoid(outputs)
            outputs = (outputs > 0.5)

        output_decoder.decode_batch(outputs.cpu().numpy().reshape(nb, x, y, z, nc).astype(np.uint8),
                                    batch_output['image_location'])



def main():

    print("[INFO]Reading data")
    # Dictionary with data parameters for NiftyNet Reader
    if torch.cuda.is_available():
        print('[INFO] GPU available.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    else:
        raise Exception("[INFO] No GPU found or Wrong gpu id, please run without --cuda")

    # Dictionary with data parameters for NiftyNet Reader
    data_param = {'image': {'path_to_search': '/home/oeslle/Documents/Datasets/CC359_NEW/data',
                            'filename_contains': 'CC'},
                  'label': {'path_to_search': '/home/oeslle/Documents/Datasets/CC359_NEW/label',
                            'filename_contains': 'CC'}}

    # NiftyNet parameters
    data_split_file = 'train_val_infer_split.csv'
    patch_size = 64
    window_border = (8,8,8)

    print("[INFO] Building model")
    in_channels = 1
    n_classes = 1
    num_epochs = 2
    lr = 1e-6
    model = cnn_utils.Modified3DUNet(in_channels, n_classes)
    criterion =  loss_utils.SoftDiceLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)


    # Partitioning dataset using NiftyNet
    image_sets_partitioner = ImageSetsPartitioner().initialise(data_param=data_param,
                                                               data_split_file=data_split_file,
                                                               new_partition=False,
                                                               ratios=[0.1, 0.1])


    dsets = {'train':dset_utils.DatasetNiftySampler(sampler=get_sampler(data_param,
                                                                        image_sets_partitioner,
                                                                        'train',
                                                                        patch_size)),
             'val':dset_utils.DatasetNiftySampler(sampler=get_sampler(data_param,
                                                                      image_sets_partitioner,
                                                                      'validation',
                                                                      patch_size))}

    # Using PytTorch DataLoader
    dataloaders = {x: DataLoader(dsets[x], batch_size=4, shuffle=True, num_workers=3)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}


    print("[INFO] Training")
    train(dataloaders,dataset_sizes,model,criterion,optimizer,num_epochs,device,in_channels)

    print("[INFO] Inference")
    inference(data_param, image_sets_partitioner, patch_size, window_border, model, device)


if __name__ == '__main__':
    main()


