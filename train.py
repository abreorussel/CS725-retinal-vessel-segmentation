import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import *
from U_Transformer import *
from transunet import *
from utils import *
from metrics import *
from config import *
import matplotlib.pyplot as plt 


cfg = Config()
train_transform = transforms.Compose([
    GrayscaleNormalization(mean=0.5, std=0.5),
    RandomFlip(),
    ToTensor(),
    Resize()
    
])


val_transform = transforms.Compose([
    GrayscaleNormalization(mean=0.5, std=0.5),
    ToTensor(),
    Resize()
])


# net = U_Transformer(1,1).to(device)
net = TransUNet(img_dim=256,
                in_channels=1,
                out_channels=128,
                head_num=4,
                mlp_dim=512,
                block_num=8,
                patch_dim=16,
                class_num=1).to(device)


print(f"train : {TRAIN_IMGS_DIR}")
# Set Dataset
train_dataset = Dataset(imgs_dir=TRAIN_IMGS_DIR, labels_dir=TRAIN_LABELS_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
val_dataset = Dataset(imgs_dir=VAL_IMGS_DIR, labels_dir=VAL_LABELS_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

train_data_num = len(train_dataset)
val_data_num = len(val_dataset)

train_batch_num = int(np.ceil(train_data_num / cfg.BATCH_SIZE)) 
val_batch_num = int(np.ceil(val_data_num / cfg.BATCH_SIZE))


# Loss Function
# loss_fn = nn.BCEWithLogitsLoss().to(device)
loss_fn = CombinedLoss(weight_dice=0.7, weight_bce=0.3).to(device)

# Optimizer
optim = torch.optim.Adam(params=net.parameters(), lr=cfg.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=5, verbose=True)


# Training
start_epoch = 0
# Load Checkpoint File
if os.listdir(CKPT_DIR):
    net, optim, start_epoch = load_net(ckpt_dir=CKPT_DIR, net=net, optim=optim)
else:
    print('* Training from scratch')

num_epochs = cfg.NUM_EPOCHS

train_loss_epoch_arr = list()
val_loss_epoch_arr = list()

for epoch in range(start_epoch+1, num_epochs+1):
    net.train()  # Train Mode
    train_loss_arr = list()

    train_batch_dice_coeff = []
    train_batch_hd95_values = []
    val_batch_dice_coeff = []
    val_batch_hd95_values = []
    
    for batch_idx, data in enumerate(train_loader, 1):
        # Forward Propagation
        img = data['img'].to(device)
        label = data['label'].to(device)
        
        output = net(img)
        
        # Backward Propagation
        optim.zero_grad()
        
        loss = loss_fn(output, label)
        loss.backward()
        
        optim.step()
        
        # Calc Loss Function
        train_loss_arr.append(loss.item())
        print_form = '[Train] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
        print(print_form.format(epoch, num_epochs, batch_idx, train_batch_num, train_loss_arr[-1]))
        
        # Tensorboard
        img = to_numpy(denormalization(img, mean=0.5, std=0.5))
        label = to_numpy(label)
        output = to_numpy(classify_class(output))

        
    train_loss_avg = np.mean(train_loss_arr)
    train_loss_epoch_arr.append(train_loss_avg)
    
    
    # Validation (No Back Propagation)
    with torch.no_grad():
        net.eval()  # Evaluation Mode
        val_loss_arr = list()
        
        for batch_idx, data in enumerate(val_loader, 1):
            # Forward Propagation
            img = data['img'].to(device)
            label = data['label'].to(device)
            
            output = net(img)
            
            # Calc Loss Function
            loss = loss_fn(output, label)
            val_loss_arr.append(loss.item())
            
            print_form = '[Validation] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
            print(print_form.format(epoch, num_epochs, batch_idx, val_batch_num, val_loss_arr[-1]))
            
            # Tensorboard
            img = to_numpy(denormalization(img, mean=0.5, std=0.5))
            label = to_numpy(label)
            output = to_numpy(classify_class(output))
        

    scheduler.step(np.sum(val_loss_arr))
            
    val_loss_avg = np.mean(val_loss_arr)
    val_loss_epoch_arr.append(val_loss_avg)
    
    print_form = '[Epoch {:0>4d}] Training Avg Loss: {:.4f} | Validation Avg Loss: {:.4f}'
    print(print_form.format(epoch, train_loss_avg, val_loss_avg))
    if epoch % 50 == 0:
      save_net(ckpt_dir=CKPT_DIR, net=net, optim=optim, epoch=epoch)


# Print the Loss Graph

import matplotlib.pyplot as plt

# Example lists of training and validation loss

epochs = range(1, len(train_loss_epoch_arr) + 1)  # Epochs (1, 2, 3, ...)

# Plot the training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss_epoch_arr, label='Training Loss', marker='o' , color="red")
plt.plot(epochs, val_loss_epoch_arr, label='Validation Loss', marker='o' , color="yellow")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
