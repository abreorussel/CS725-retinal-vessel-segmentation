import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import cv2

from torchvision import transforms
from dataset import *
from torch.utils.data import DataLoader
# from U_Transformer import *
from transunet import *
from metrics import *
# from model import UNet
from utils import *
from config import *

def construct_overlay_image(img_path, output_path, label_path , crt_id):
    # Load original image and the masks (segmented and ground truth)
    original_img = cv2.imread(img_path)  # Load the original image
    segmented_img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)  # Predicted binary mask (0, 255)
    gt_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Ground truth binary mask (0, 255)

    # Ensure the original image is in RGB format (OpenCV loads as BGR by default)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Define colors for overlay
    # Considereing RGB
    prediction_color = [58, 140, 255]  # Blue for the predicted mask
    gt_color = [47, 237, 57]  # Green for the ground truth mask
    intersection_color = [236, 255, 88]  # Yellow for the intersection of both masks

    # Create empty colored masks
    colored_prediction = np.zeros_like(original_img)
    colored_gt = np.zeros_like(original_img)
    colored_intersection = np.zeros_like(original_img)

    # Create masks for prediction, ground truth, and their intersection
    prediction_mask = segmented_img == 255
    gt_mask = gt_img == 255
    intersection_mask = np.logical_and(prediction_mask, gt_mask)

    # Apply colors to each region
    colored_prediction[prediction_mask] = prediction_color  # Red for prediction
    colored_gt[gt_mask] = gt_color  # Yellow for ground truth
    colored_intersection[intersection_mask] = intersection_color  # Green for intersection

    # Copy the original image and overlay the colored masks
    overlay_img = original_img.copy()

    # Blend the original image with the predicted and ground truth masks
    overlay_img[prediction_mask] = cv2.addWeighted(original_img[prediction_mask], 0.6, colored_prediction[prediction_mask], 0.6, 0)
    overlay_img[gt_mask] = cv2.addWeighted(original_img[gt_mask], 0.6, colored_gt[gt_mask], 0.6, 0)
    overlay_img[intersection_mask] = cv2.addWeighted(original_img[intersection_mask], 0.6, colored_intersection[intersection_mask], 0.6, 0)

    # Save the resulting overlay image
    cv2.imwrite(os.path.join(RESULTS_DIR, f'overlay_{crt_id:04}.png'), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))



cfg = Config()
transform = transforms.Compose([
    GrayscaleNormalization(mean=0.5, std=0.5),
    ToTensor(),
    Resize()
])

RESULTS_DIR = os.path.join(ROOT_DIR, 'test_results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

test_dataset = Dataset(imgs_dir=TEST_IMGS_DIR, labels_dir=TEST_LABELS_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

test_data_num = len(test_dataset)
test_batch_num = int(np.ceil(test_data_num / cfg.BATCH_SIZE)) # np.ceil 반올림

# Network
# net = U_Transformer(1,1).to(device)
net = TransUNet(img_dim=256,
                in_channels=1,
                out_channels=128,
                head_num=4,
                mlp_dim=512,
                block_num=8,
                patch_dim=16,
                class_num=1).to(device)

# Loss Function
loss_fn = nn.BCEWithLogitsLoss().to(device)

# Optimizer
optim = torch.optim.Adam(params=net.parameters(), lr=cfg.LEARNING_RATE)

start_epoch = 0

# Load Checkpoint File
if os.listdir(CKPT_DIR):
    net, optim, _ = load_net(ckpt_dir=CKPT_DIR, net=net, optim=optim)

# Evaluation
with torch.no_grad():
    net.eval()  # Evaluation Mode
    loss_arr = list()
    dice_arr = list()
    hd95_arr = list()

    for batch_idx, data in enumerate(test_loader, 1):
        # Forward Propagation
        img = data['img'].to(device)
        label = data['label'].to(device)

        output = net(img)   # Performing forward pass over the entire batch

        # Calc Loss Function
        loss = loss_fn(output, label)
        loss_arr.append(loss.item())

        # Tensorboard
        img = to_numpy(denormalization(img, mean=0.5, std=0.5))
        label = to_numpy(label)
        output = to_numpy(classify_class(output))


        batch_dice_coeff = []
        batch_hd95_values = []

        for j in range(label.shape[0]):
            crt_id = int(test_batch_num * (batch_idx - 1) + j)

            single_output = output[j].squeeze()  # Shape: [1, channels, height, width]
            single_label = label[j].squeeze()

            # single_output = cv2.imread("label_0000.png" , cv2.IMREAD_GRAYSCALE)

            dice_coeff = dc(single_output , single_label)
            hd95_value = hd95(single_output , single_label)
            # print(f"crt : {crt_id} dice : {dice_coeff} hd_95 : {hd95_value}")

            batch_dice_coeff.append(dice_coeff)
            batch_hd95_values.append(hd95_value)

            image_path = os.path.join(RESULTS_DIR, f'img_{crt_id:04}.png')
            output_path = os.path.join(RESULTS_DIR, f'output_{crt_id:04}.png')
            label_path = os.path.join(RESULTS_DIR, f'label_{crt_id:04}.png')

            plt.imsave(os.path.join(RESULTS_DIR, f'img_{crt_id:04}.png'), img[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(RESULTS_DIR, f'label_{crt_id:04}.png'), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(RESULTS_DIR, f'output_{crt_id:04}.png'), output[j].squeeze(), cmap='gray')
            construct_overlay_image(image_path, output_path ,label_path, crt_id)

        batch_mean_dice = np.mean(batch_dice_coeff)
        batch_mean_hd95 = np.mean(batch_hd95_values)
        hd95_arr.append(batch_mean_hd95)
        dice_arr.append(batch_mean_dice)
        print_form = '[Test] | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f} | Dice: {:.4f} | hd95: {:.4f}'
        print(print_form.format(batch_idx, test_batch_num, loss_arr[-1] , batch_mean_dice , batch_mean_hd95))

print_form = '[Result] | Avg Loss: {:0.4f} | Avg Dice Coeff : {:0.4f} | Avg hd95 : {:.4f}'
print(print_form.format(np.mean(loss_arr) , np.mean(dice_arr) , np.mean(hd95_arr)))
