
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import os
from time import localtime, strftime
import yaml
import argparse
import time
import copy

from model import *
from data import *

# MRAugment-specific imports
from mraugment.data_augment import DataAugmentor
from mraugment.data_transforms import UnetDataTransform

parser = argparse.ArgumentParser(description='CS7643 Final Project GAN')
parser.add_argument('--config', default='config.yaml')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def train(device, epoch, data_loader, Generater_obj, Discriminator_obj, Goptimizer, Doptimizer, criterion1, criterion2):
    """
    criterion1 = BCELoss
    criterion2 = MSELoss
    """
    global coeff_adv, coeff_pw

    iter_time = AverageMeter()
    Dis_losses = AverageMeter()
    Gen_losses = AverageMeter()

    for idx, data in enumerate(data_loader):
        start = time.time()
        sampled, target, mean, std = data[0], data[1], data[2], data[3]
        # if torch.cuda.is_available():
        sampled = sampled.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        fake_img = Generater_obj(sampled).to(device)

        # train discrimiator
        d_real_logit = Discriminator_obj(target)
        real_loss = criterion1(d_real_logit, torch.ones(d_real_logit.shape).to(device))  # label 1 as real
        d_fake_logit = Discriminator_obj(fake_img)
        fake_loss = criterion1(d_fake_logit, torch.zeros(d_fake_logit.shape).to(device)) # label 0 as fake

        Disloss = real_loss + fake_loss
        Doptimizer.zero_grad()
        Disloss.backward(retain_graph=True)
        Doptimizer.step()

        # train generator
        g_fake_logit = Discriminator_obj(fake_img)
        gen_adv_loss = criterion1(g_fake_logit, torch.ones(g_fake_logit.shape).to(device))
        gen_pw_loss = criterion2(fake_img, target)  # pixel wise loss
        Genloss = gen_adv_loss * coeff_adv + gen_pw_loss * coeff_pw
        Goptimizer.zero_grad()
        Genloss.backward(retain_graph=True)
        Goptimizer.step()

        Dis_losses.update(Disloss, sampled.shape[0])
        Gen_losses.update(Genloss, sampled.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Discriminator Loss {Dis_losses.val:.4f} ({Dis_losses.avg:.4f})\t'
                   'Generator Loss {Gen_losses.val:.4f} ({Gen_losses.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, Dis_losses=Dis_losses, Gen_losses=Gen_losses))

def validate(device, epoch, data_loader, Generator_obj, Discriminator_obj, criterion1, criterion2):
    """
    criterion1 = BCELoss
    criterion2 = MSELoss
    """
    global coeff_adv, coeff_pw

    iter_time = AverageMeter()
    Dis_losses = AverageMeter()
    Gen_losses = AverageMeter()

    Generator_obj.eval()
    Discriminator_obj.eval()

    for idx, data in enumerate(data_loader):
        start = time.time()
        sampled, target, mean, std = data[0], data[1], data[2], data[3]
        # if torch.cuda.is_available():
        sampled = sampled.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        with torch.no_grad():
            fake_img = Generator_obj(sampled).to(device)
            d_real_logit = Discriminator_obj(target)
            real_loss = criterion1(d_real_logit, torch.ones(d_real_logit.shape).to(device))
            d_fake_logit = Discriminator_obj(fake_img)
            fake_loss = criterion1(d_fake_logit, torch.zeros(d_fake_logit.shape).to(device))
            Disloss = real_loss + fake_loss

            g_fake_logit = Discriminator_obj(fake_img)
            gen_adv_loss = criterion1(g_fake_logit, torch.ones(g_fake_logit.shape).to(device))
            gen_pw_loss = criterion2(fake_img, target)  # pixel wise loss
            Genloss = gen_adv_loss * coeff_adv + gen_pw_loss * coeff_pw

            iter_time.update(time.time() - start)
            if idx % 10 == 0:
                print(('Epoch: [{0}][{1}/{2}]\t'
                    'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                    'Discriminator Loss {Dis_losses.val:.4f} ({Dis_losses.avg:.4f})\t'
                    'Generator Loss {Gen_losses.val:.4f} ({Gen_losses.avg:.4f})\t')
                    .format(epoch, idx, len(data_loader), iter_time=iter_time, Dis_losses=Dis_losses, Gen_losses=Gen_losses))
            
        Dis_losses.update(Disloss, sampled.shape[0])
        Gen_losses.update(Genloss, sampled.shape[0])

    return Dis_losses.avg, Gen_losses.avg



def main():
    global args, coeff_adv, coeff_pw, epoch
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coeff_adv, coeff_pw = args.coeff_adv, args.coeff_pw
    
    # Data augmentation
    aug_args = {
        'accelerations': [8],
        'aug_delay': 0,
        'aug_exp_decay': 5.0,
        'aug_interpolation_order': 1,
        'aug_max_rotation': 180.0,
        'aug_max_scaling': 0.25,
        'aug_max_shearing_x': 15.0,
        'aug_max_shearing_y': 15.0,
        'aug_max_translation_x': 0.125,
        'aug_max_translation_y': 0.08,
        'aug_on': True,
        'aug_schedule': 'exp',
        'aug_strength': 0.55,
        'aug_upsample': True,
        'aug_upsample_factor': 2,
        'aug_upsample_order': 1,
        'aug_weight_fliph': 0.5,
        'aug_weight_flipv': 0.5,
        'aug_weight_rot90': 0.5,
        'aug_weight_rotation': 0.5,
        'aug_weight_scaling': 1.0,
        'aug_weight_shearing': 1.0,
        'aug_weight_translation': 1.0,
        'max_train_resolution': None,
        'batch_size': 1,
        'center_fractions': [0.04],
        'challenge': "singlecoil",
        'chans': 18,
        'check_val_every_n_epoch': 10,
        'lr': 0.0003,
        'lr_gamma': 0.1,
        'lr_step_size': 500,
        'mask_type': "random",
        'max_epochs': 10,
        'num_cascades': 12,
        'pools': 4,
        'resume_from_checkpoint': None,
        'seed': 42,
        'train_resolution': [640, 368],
        'volume_sample_rate': 0.1,
        'weight_decay': 0.0,
    }
    
    # returns the current epoch for p scheduling
    current_epoch_fn = lambda: globals()["epoch"]
    
    # initialize data augmentation pipeline
    augmentor = DataAugmentor(aug_args, current_epoch_fn)
    
    train_dataset = mri_data.SliceDataset(
#         root=pathlib.Path('./data/singlecoil_train/'),
        root=pathlib.Path('/mnt/e/fastMRI/singlecoil_val/'),
        transform=UnetDataTransform(which_challenge="singlecoil", augmentor=augmentor),
        challenge='singlecoil'
    )
    val_dataset = mri_data.SliceDataset(
#         root=pathlib.Path('./data/singlecoil_val/'),
        root=pathlib.Path('/mnt/e/fastMRI/singlecoil_val/'),
        transform=UnetDataTransform(which_challenge="singlecoil", augmentor=augmentor),
        challenge='singlecoil'
    )
    
#     train_dataset = mri_data.SliceDataset(
#         root=pathlib.Path('./data/singlecoil_train/'),
#         transform=UnetDataTransform(which_challenge="singlecoil"),
#         challenge='singlecoil'
#     )
#     val_dataset = mri_data.SliceDataset(
#         root=pathlib.Path('./data/singlecoil_val/'),
#         transform=UnetDataTransform(which_challenge="singlecoil"),
#         challenge='singlecoil'
#     )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    Generater_model = Generater().to(device)
    Discriminator_model = Discriminator().to(device)

    Doptimizer = torch.optim.SGD(Discriminator_model.parameters(), args.d_learning_rate,
                                 momentum=args.d_momentum,
                                 weight_decay=args.d_reg)
    Goptimizer = torch.optim.SGD(Generater_model.parameters(), args.g_learning_rate,
                                 momentum=args.g_momentum,
                                 weight_decay=args.g_reg)

    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    best_dis_loss = float("Inf")
    best_gen_loss = float("Inf")
    best_dis_model = None
    best_gen_model = None
    for epoch in range(args.epochs):
        globals()["epoch"] = epoch
        # adjust_learning_rate(Doptimizer, epoch, args)
        # adjust_learning_rate(Goptimizer, epoch, args)

        # train loop
        train(device, epoch, train_loader, Generater_model, Discriminator_model, Goptimizer, Doptimizer, criterion_bce, criterion_mse)

        # validation loop
        dis_loss, gen_loss = validate(device, epoch, val_loader, Generater_model, Discriminator_model, criterion_bce, criterion_mse)

        if dis_loss < best_dis_loss:
            best_dis_loss = dis_loss
            best_dis_model = copy.deepcopy(Discriminator_model)

        if gen_loss < best_gen_loss:
            best_gen_loss = gen_loss
            best_gen_model = copy.deepcopy(Generater_model)

    if args.save_best:
        torch.save(best_gen_model.state_dict(), './checkpoints/' + 'generator' + '.pth')
        torch.save(best_dis_model.state_dict(), './checkpoints/' + 'discriminator' + '.pth')
        
    torch.save(Generater_model.state_dict(), './checkpoints/' + 'final_generator' + '.pth')
    torch.save(Discriminator_model.state_dict(), './checkpoints/' + 'final_discriminator' + '.pth')

if __name__ == '__main__':
    main()
