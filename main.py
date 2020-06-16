from __future__ import print_function
import os
import sys
sys.path.append('../../')
import argparse
import yaml
import time
import tqdm
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import Utils
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from Utils.loss import ReverseHuberLoss #as ReverseHuberLoss
from termcolor import colored
from itertools import chain

#torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Training script for 360 layout',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', default='test', type=str, help='train/test mode')
parser.add_argument('--pre', default=None, type=int, help='pretrain(default: latest)')
parser.add_argument('--log', default='Results.txt', type=str, help='log file name')
args = parser.parse_args()


def train(args, config, dataset_train, dataset_val, model, saver, writer):
    # Summary writer
    #writer = SummaryWriter(config['exp_path'])


    [_, offset] = saver.LoadLatestModel(model, args.pre)
    finish_epoch = offset
    offset = offset * len(dataset_train)
    param = chain(model.conv_mask.parameters(), model.conv_e2c.parameters(), model.conv_c2e.parameters())
    #param = model.parameters()
    optim = torch.optim.Adam(param, lr=config['lr'])
    #optim = torch.optim.SGD(model.parameters(), lr=config['lr'],decay=config[''])
    model = nn.DataParallel(model)
    start_epoch = 0
    for epoch in range(finish_epoch, config['epochs']):
    	# update learning rate
        #schedular.step() if SGD
        offset = train_an_epoch(config, model, dataset_train, optim, writer, epoch, offset)
        saver.Save(model.module, epoch)

        global_step = (epoch + 1) * len(dataset_train)
        val_results = val_an_epoch(dataset_val, model, config, writer)
        print(colored('\nDense Results: ', 'magenta'))
        for name, val in val_results['dense-equi'].items():
            print(colored('- {}: {}'.format(name, val), 'magenta'))
            writer.add_scalar('C-val-dense-metric/{}'.format(name), val, epoch)
        with open(args.log, 'a') as f:
            f.write('This is %d epoch:\n'%(epoch))
            for name, val in val_results['dense-equi'].items():
                f.write('--- %s: %f\n'%(name, val))
            f.close()

def train_an_epoch(config, model, loader, optim, writer, epoch, step_offset):
    model.train()
    meters = Utils.Metrics(['rmse', 'mae', 'mre'])
    #ReverseHuberLoss() = Utils.ReverseHuberLoss()
    criterion_dict = dict()
    criterion_dict = {'ReverseHuberLoss': ReverseHuberLoss()}
    berhu = ReverseHuberLoss()
    iter_time = 0
    loss = 0
    #it = 0
    CE = Utils.CETransform()
    grid = Utils.Equirec2Cube(None, 512, 1024, 256, 90).GetGrid()
    d2p = Utils.Depth2Points(grid)

    for i, data in enumerate(loader):
        tic = time.time()
        it = i + step_offset
        raw_rgb_var, rgb_var, depth_var = data['raw_rgb'], data['rgb'], data['depth']
        raw_rgb_var = raw_rgb_var.cuda()
        rgb_var = rgb_var.cuda()
        depth_var = depth_var.cuda()
        
        rgb_equi = rgb_var

        pred_var, pred_cube_var = model(rgb_equi)
        #'''
        cube_pts = d2p(pred_cube_var)
        pred_cube_var = CE.C2E(torch.norm(cube_pts, p=2, dim=3).unsqueeze(1))
        #'''
        total_loss_var = 0
        loss_var_dict = dict()
        loss_var_dict['BerHu-equi'] = berhu(pred_var, depth_var)
        loss_var_dict['BerHu-cube'] = berhu(pred_cube_var, depth_var)
        total_loss_var = loss_var_dict['BerHu-equi'] + loss_var_dict['BerHu-cube']

        optim.zero_grad()
        total_loss_var.backward()
        optim.step()

        toc = time.time()
        iter_time += (toc - tic)
        loss += float(total_loss_var.cpu())
        #it = 0
        if it % config['print_step'] == 0:
            pred, depth = pred_var.data.cpu().numpy(), depth_var.data.cpu().numpy()
            results = meters.compute(pred, depth)
            iter_time /= config['print_step']
            loss /= config['print_step']
            print('[{}/{}][{}/{}] loss:{:.3f} time:{:.2f} '.format(epoch, config['epochs'] - 1, it % len(loader),
                                                                   len(loader) - 1, loss, iter_time), end='')
            
            for name, val in results.items():
                print('{}:{:.3f} '.format(name, val), end='')
            
            print('')
            iter_time = 0
            loss = 0
            
        # log results to tensorboard
        if it % config['log_step'] == 0:
            pred, depth = pred_var.data.cpu().numpy(), depth_var.data.cpu().numpy()
            results = meters.compute(pred, depth)
            global_step = epoch * len(loader) + i
            writer.add_scalar('A-loss/total_loss', total_loss_var.data, global_step)
            for name, loss_var in loss_var_dict.items():
                writer.add_scalar('A-loss/{}'.format(name), loss_var.data, global_step)
            for name, val in results.items():
                writer.add_scalar('B-train-dense-metric/{}'.format(name), val, global_step)
            rgb_grid = vutils.make_grid(raw_rgb_var, nrow=8, normalize=True, scale_each=True)
            writer.add_image('A-rgb', rgb_grid, global_step)
            writer.add_image('B-prediction-equi', torch.clamp(pred_var, 0, 10)/10, global_step)
            writer.add_image('D-depth', torch.clamp(depth_var, 0, 10)/10, global_step)
    return it
def val_an_epoch(loader, model, config, writer):
    model = model.eval()
    #meters = Utils.Metrics(['rmse', 'mae', 'mre'])
    meters = Utils.Metrics(config['metrics'])
    avg_meters = Utils.MovingAverageEstimator(config['metrics'])
    avg_meters_cube = Utils.MovingAverageEstimator(config['metrics'])
    pbar = tqdm.tqdm(loader)
    pbar.set_description('Validation process')
    #pbar = loader
    gpu_num = torch.cuda.device_count()

    CE = Utils.CETransform()
    grid = Utils.Equirec2Cube(None, 512, 1024, 256, 90).GetGrid()
    d2p = Utils.Depth2Points(grid)

    with torch.no_grad():
        for it, data in enumerate(pbar):
            raw_rgb_var, rgb_var, depth_var = data['raw_rgb'], data['rgb'], data['depth']
            raw_rgb_var, rgb_var, depth_var = raw_rgb_var.cuda(), rgb_var.cuda(), depth_var.cuda()

            inputs = rgb_var
            if inputs.shape[0] % gpu_num == 0:
                raw_pred_var, pred_cube_var = model(inputs)
            else:
                raw_pred_var = []
                pred_cube_var = []
                count = inputs.shape[0] // gpu_num
                lf = inputs.shape[0] % gpu_num
                for gg in range(count):
                    a = inputs[gg*gpu_num:(gg+1)*gpu_num]
                    a, b = model(a)
                    raw_pred_var.append(a)
                    pred_cube_var.append(b)

                a = inputs[count*gpu_num:]
                a, b = model.module(a)
                raw_pred_var.append(a)
                pred_cube_var.append(b)
                raw_pred_var = torch.cat(raw_pred_var, dim=0)
                pred_cube_var = torch.cat(pred_cube_var, dim=0)
            #'''
            cube_pts = d2p(pred_cube_var)
            pred_cube_var = CE.C2E(torch.norm(cube_pts, p=2, dim=3).unsqueeze(1))
            #'''
            for i in range(raw_pred_var.shape[0]):
                pred = raw_pred_var[i:i+1].data.cpu().numpy()
                pred_cube = pred_cube_var[i:i+1].data.cpu().numpy()
                depth = depth_var[i:i+1].data.cpu().numpy()

                results = meters.compute(pred.clip(0, 10), depth.clip(0, 10))
                results_cube = meters.compute(pred_cube.clip(0, 10), depth.clip(0, 10))
                avg_meters.update(results) 
                avg_meters_cube.update(results_cube)
                #print (results)
                #print (results_cube)
           
    # Print final results and log to tensorboard
    final_results = {
        'dense-equi': avg_meters.compute(),
        'dense-cube': avg_meters_cube.compute()
    }
    
 
    #print('')
    #model = model.train()
    return final_results

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
        print (json.dumps(config, indent=4))

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 
    from Dataset_mp3d_npy import Dataset
    pano_dataset_train = Dataset(config['root_path'], mode='train_full')
    pano_dataset_val = Dataset(config['root_path'], mode='val_full')


    dataset_train = DataLoader(
            pano_dataset_train, 
            batch_size=config['batch_size'],
            num_workers=config['processes'],
            drop_last=True,
            pin_memory=True,
            shuffle=True
            )
    dataset_val = DataLoader(
            pano_dataset_val, 
            batch_size=config['batch_size'],
            num_workers=config['processes'],
            drop_last=False,
            pin_memory=True,
            shuffle=False
            )

    saver = Utils.ModelSaver(config['save_path'])
    from models.FCRN import MyModel as ResNet
    model = ResNet(
    		layers=config['model_layer'],
    		decoder=config['decoder_type'],
    		output_size=None,
    		in_channels=3,
    		pretrained=True
    		).cuda()
    if args.mode == 'train':
        #writer = Utils.visualizer(config['exp_path'])
        writer = SummaryWriter(config['exp_path'])
        train(args, config, dataset_train, dataset_val, model, saver, writer)
    else:
        saver.LoadLatestModel(model, args.pre)

        writer = None
        model = nn.DataParallel(model)
        results = val_an_epoch(dataset_val, model, config, writer)
        print(colored('\nDense Results: ', 'magenta'))
        for name, val in results['dense-equi'].items():
            print(colored('- {}: {}'.format(name, val), 'magenta'))
        for name, val in results['dense-cube'].items():
            print(colored('- {}: {}'.format(name, val), 'magenta'))

        #results = val_an_epoch(model, dataset_val, 0)


if __name__ == '__main__':
    main()
