
from dataset import GratingDataset
from transforms import GaussianNoise
from alexnet_rnn import AlexNetRNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import models, transforms
from utils import *
import wandb

import numpy as np
import math
import random

if __name__ == '__main__':

    data_transforms = transforms.Compose([
        transforms.Resize(227), # changed from 128
        transforms.ToTensor(),
        GaussianNoise(0, 0.01), # STANDARD DEVIATION OF GAUSSIAN NOISE
    ])
    
    wandb.login()

    root_dir = './SG_train_double_sf/'
    test_root_dir = './SG_test_double_sf/'
    
    num_seqs = 1000
    batch_size = 100
    num_epochs = 30

    num_workers = 8 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dir_list = [
                [root_dir+'sep_10.0',0,0.05,10.0],
                [root_dir+'sep_5.0',0,0.05,5.0],
                [root_dir+'sep_2.0',0,0.05,2.0],
                [root_dir+'sep_1.0',0,0.05,1.0],
                [root_dir+'sep_0.5',0,0.05,0.5]
                ]
    dir_list.reverse()

    test_dir_list = [
                    [test_root_dir+'sep_10.0',0,0.1,10.0],
                    [test_root_dir+'sep_5.0',0,0.1,5.0],
                    [test_root_dir+'sep_2.0',0,0.1,2.0],
                    [test_root_dir+'sep_1.0',0,0.1,1.0],
                    [test_root_dir+'sep_0.5',0,0.1,0.5]
                    ]
    test_dir_list.reverse()

    models = []
    for i in range(1):

        train_dir = dir_list[i]
        train_root_dir = train_dir[0]
        train_ref_ori = train_dir[1]
        train_sf = train_dir[2]
        train_sep = train_dir[3]

        train_ref_dir = './SG_refs/' + 'REFERENCE_ref_'+str(train_ref_ori)+'_sep_0.0_contr_1_ph_0.0_sf_'+str(train_sf)+'_NONE.png'

        train_grating_dataset = GratingDataset(train_root_dir, train_ref_dir, transform=data_transforms, num_seqs=num_seqs)
        train_dataloader = DataLoader(train_grating_dataset, batch_size=batch_size, shuffle=True, num_workers= num_workers)


        test_dir = test_dir_list[i]

        test_root_dir = test_dir[0]
        test_ref_ori = test_dir[1]
        test_sf = test_dir[2]
        test_sep = test_dir[3]
        test_ref_dir = './SG_refs/'+ 'REFERENCE_ref_' + str(test_ref_ori)+'_sep_0.0_contr_1_ph_0.0_sf_'+str(test_sf)+'_NONE.png'
        test_grating_dataset = GratingDataset(test_root_dir, test_ref_dir, transform=data_transforms, num_seqs=num_seqs)
        test_dataloader = DataLoader(test_grating_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        wandb.init(
            project="Alex-GRU-BCE-MSE-pred-new",
            name="sep-{}".format(train_sep),
            settings=wandb.Settings(start_method="fork"),
            config={
                "epochs": num_epochs,
                "batch_size": batch_size,
                "lr": 5e-3,
                "sep": train_sep,
                "pred_mult": .01
        })

          # Copy your config
        config = wandb.config
        n_steps_per_epoch = math.ceil(len(train_dataloader.dataset) / config.batch_size)

        alexnet = torchvision.models.alexnet(pretrained=True)
        model= AlexNetRNN()
        copy_weights(model, alexnet)
        model.to(device)

        loss_fn = nn.BCELoss()
        loss_fn2 = nn.MSELoss()

        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.8)

        example_ct = 0
        step_ct = 0

        for epoch in range(config.epochs):
            
            init_train_loss1, init_train_loss2, accuracy = validate_model(model, train_dataloader, loss_fn, loss_fn2, device=device)

            metrics = {"train/BCE_Loss": init_train_loss1,
            "train/MSE_loss": init_train_loss2,
            "train/train_loss": init_train_loss1 + config["pred_mult"]*init_train_loss2,
            "train/epoch": 0,
            "train/example_ct": example_ct,
            "train/train_accuracy": accuracy}


            model.train()
            for step, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(device), labels.to(device)
                # labels = labels.squeeze()

                outputs, rnn_input, pred_out = model(images)
                last_outputs = outputs[:,-5:]

                optimizer.zero_grad()

                loss1 = loss_fn(last_outputs, labels)
                loss2 = loss_fn2(rnn_input, pred_out) 
                # loss2 = loss_fn2(rnn_input, pred_out) + compute_var(pred_out)

                train_loss = loss1 + config["pred_mult"]*loss2

                # train_loss = loss1

                train_loss.backward()

                optimizer.step()

                example_ct += len(images)

                num_corrects = calculate_corrects(outputs[:,-1], labels[:,-1])

                metrics = {"train/train_loss": train_loss,
                            "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                            "train/example_ct": example_ct,
                            "train/BCE_Loss": loss1, 
                            "train/MSE_loss": loss2, 
                            "train/train_accuracy": num_corrects/len(images)}

                if step + 1 < n_steps_per_epoch:
                    # 🐝 Log train metrics to wandb
                    wandb.log(metrics)

                step_ct += 1

            val_loss1, val_loss2, accuracy = validate_model(model, test_dataloader, loss_fn, loss_fn2, device=device)

            # 🐝 Log train and validation metrics to wandb
            val_metrics = {"val/BCE_Loss": val_loss1,
                           "val/MSE_loss": val_loss2,
                            "val/val_accuracy": accuracy}
            wandb.log({**metrics, **val_metrics})

            # print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")
            if epoch%5 ==0:
                torch.save(model, 'data/output/model_sep{}_epoch{}.pth'.format(train_sep, epoch))


         # If you had a test set, this is how you could log it as a Summary metric
         # 🐝 Close your wandb run
        torch.save(model, 'data/output/model_sep{}_last_epoch.pth'.format(train_sep))
        wandb.finish()

