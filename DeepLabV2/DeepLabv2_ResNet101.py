# %%
import os
import zipfile

# URL for the dataset
url = "https://zenodo.org/records/5706578/files/Train.zip?download=1"

# Download the file using wget
!wget -O /content/Train.zip "$url"

# Define the extraction path
extract_path = '/content/datasets/Train/'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile('/content/Train.zip', 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# List the contents of the extracted folder
extracted_files = os.listdir(extract_path)
print("Extracted files:", extracted_files)


# %%
import os
import zipfile

# URL for the dataset
url = "https://zenodo.org/records/5706578/files/Val.zip?download=1"

# Download the file using wget
!wget -O /content/Val.zip "$url"

# Define the extraction path
extract_path = '/content/datasets/Val/'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile('/content/Val.zip', 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# List the contents of the extracted folder
extracted_files = os.listdir(extract_path)
print("Extracted files:", extracted_files)


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models

expansion = 4

class ConvBN(nn.Module):  # Convolutional followed by Batch Norm
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=1e-3)

    def forward(self, x):
        return self.bn(self.conv(x))

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, downsample=False, dropout_rate=0.5):
        super(Bottleneck, self).__init__()
        mid_planes = out_planes // expansion
        self.conv1 = ConvBN(in_planes, mid_planes, kernel_size=1, stride=stride)
        self.relu1 = nn.ReLU(inplace=True)

        # Dropout after the first convolution
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.conv2 = ConvBN(mid_planes, mid_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.relu2 = nn.ReLU(inplace=True)

        # Dropout after the second convolution
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv3 = ConvBN(mid_planes, out_planes, kernel_size=1)
        self.relu3 = nn.ReLU(inplace=True)

        # Dropout after the third convolution
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        if downsample:
            self.shortcut = ConvBN(in_planes, out_planes, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu1(self.conv1(x))

        # Apply dropout after the first convolution
        out = self.dropout1(out)

        out = self.relu2(self.conv2(out))

        # Apply dropout after the second convolution
        out = self.dropout2(out)

        out = self.conv3(out)

        # Apply dropout after the third convolution
        out = self.dropout3(out)

        out += identity
        return self.relu3(out)

def make_layer(blocks, in_planes, out_planes, stride, dilation, dropout_rate=0.5):
    layers = OrderedDict()
    layers['block1'] = Bottleneck(in_planes, out_planes, stride=stride, dilation=dilation, downsample=True, dropout_rate=dropout_rate)
    for i in range(1, blocks):
        layers[f'block{i+1}'] = Bottleneck(out_planes, out_planes, stride=1, dilation=dilation, dropout_rate=dropout_rate)
    return nn.Sequential(layers)

class ASPP(nn.Module):
    def __init__(self, in_planes, out_planes, atrous_rates):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                      padding=rate, dilation=rate, bias=True) for rate in atrous_rates
        ])
        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            nn.init.normal_(conv.weight, mean=0, std=0.01)
            nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        return sum(conv(x) for conv in self.convs)


# Define DeepLabV2 Model
class DeepLabV2(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabV2, self).__init__()

        from torchvision.models import ResNet101_Weights
        model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)


        # Keep only layers up to layer4
        self.backbone = nn.Sequential(*(list(model.children())[:-2]))  # Exclude the final FC layer
        self.aspp = nn.ModuleList([
            nn.Conv2d(2048, 256, kernel_size=3, padding=r, dilation=r, bias=True) # list of modules with different dilation rates
            for r in [6, 12, 18, 24]
        ])
        self.classifier = nn.Conv2d(256, n_classes, kernel_size=1)

        # Add upsampling layer
        #self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True) # Upsample by 32 to match input size
        self.upsample = nn.Upsample(size=(720, 720), mode='bilinear', align_corners=True)  # Match target size

    def forward(self, x):
        x = self.backbone(x)
        aspp_out = sum(aspp(x) for aspp in self.aspp) # the outputs of the four convolutions are summed together

        x = self.classifier(aspp_out) # Apply the classifier

        # Upsample the output
        x = self.upsample(x) # Apply upsampling

        return x


# %%
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
!pip install thop
from thop import profile
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Replace with your dataset class and helper functions
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

class SimpleSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, preferred_resolution=(720, 720), original_resolution=(1024, 1024), transform=None, augment=False, validation=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.preferred_resolution = preferred_resolution
        self.original_resolution = original_resolution
        self.transform = transform
        self.augment = augment
        self.validation = validation
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        # Training augmentation transforms
        self.aug_transform = A.Compose([
            A.Resize(height=self.preferred_resolution[0], width=self.preferred_resolution[1], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        # Validation transform (original resolution)
        self.val_transform = A.Compose([
            A.Resize(height=self.original_resolution[0], width=self.original_resolution[1], p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx]))

        # Convert image and mask to numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        if self.validation:
            # Apply validation transform for original resolution
            transformed = self.val_transform(image=image, mask=mask)
        else:
            # Apply augmentation transform for training
            transformed = self.aug_transform(image=image, mask=mask)

        image = transformed["image"]
        mask = transformed["mask"]

        # Ensure mask is a LongTensor for CrossEntropyLoss
        mask = mask.clone().detach().to(dtype=torch.long)

        return image, mask


# Define Transform for Validation
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Dice Loss Implementation
class DiceLossIgnoringIndex0(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLossIgnoringIndex0, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        if preds.shape[1] > 1:
            preds = F.softmax(preds, dim=1)
        num_classes = preds.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        preds = preds[:, 1:]
        targets_one_hot = targets_one_hot[:, 1:]
        intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))
        union = torch.sum(preds, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        dice_score = (2.0 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - dice_score.mean()
        return loss



# Calculate IoU (Intersection over Union) for validation
def calculate_iou(output, target, num_classes):
    output = torch.argmax(output, dim=1)
    iou_list = []
    for i in range(num_classes):
        intersection = ((output == i) & (target == i)).sum().float()
        union = ((output == i) | (target == i)).sum().float()
        iou = intersection / (union + 1e-6)  # Avoid division by zero
        iou_list.append(iou.item())
    return np.mean(iou_list)


def calculate_iou_ignore_index_0(output, target, num_classes):
    """
    Calculate mean IoU (mIoU) for each class, ignoring class index 0.

    Args:
    - output (Tensor): The predicted output (batch_size, height, width)
    - target (Tensor): The ground truth target mask (batch_size, height, width)
    - num_classes (int): The number of classes in the segmentation task

    Returns:
    - (float): The mean IoU over all classes, excluding class index 0.
    """
    output = torch.argmax(output, dim=1)
    iou_list = []

    for i in range(1, num_classes):  # Start from 1 to ignore index 0
        intersection = ((output == i) & (target == i)).sum().float()
        union = ((output == i) | (target == i)).sum().float()
        iou = intersection / (union + 1e-6)  # Avoid division by zero
        iou_list.append(iou.item())

    return np.mean(iou_list) if iou_list else 0.0


import matplotlib.pyplot as plt

def visualize_predictions(images, predictions, ground_truths, num_classes):
    for idx, (image, pred, gt) in enumerate(zip(images, predictions, ground_truths)):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())  # Original input image
        plt.subplot(1, 3, 2)
        plt.title("Prediction")
        plt.imshow(pred.cpu().numpy(), cmap='tab20', vmin=0, vmax=num_classes-1)  # Prediction mask
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(gt.cpu().numpy(), cmap='tab20', vmin=0, vmax=num_classes-1)  # Actual ground truth mask

from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(preferred_resolution=(512, 512)):
    # Paths and Hyperparameters
    dataset_dir = "datasets/Train/Train/Rural"
    output_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    log_dir = "logs"
    batch_size = 6
    num_classes = 8
    lr = 0.001
    epochs = 20
    save_interval = 5

    # Dataset Paths
    train_images = os.path.join(dataset_dir, "images_png")
    train_masks = os.path.join(dataset_dir, "masks_png")
    val_dir = "datasets/Val/Val/Rural"
    val_images = os.path.join(val_dir, "images_png")
    val_masks = os.path.join(val_dir, "masks_png")

    # Datasets and DataLoaders
    train_dataset = SimpleSegmentationDataset(train_images, train_masks, preferred_resolution=preferred_resolution, augment=True)
    val_dataset = SimpleSegmentationDataset(val_images, val_masks, original_resolution=(1024, 1024), augment=False, validation=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model Initialization
    model = DeepLabV2(n_classes=num_classes)
    model = nn.DataParallel(model).cuda()

    # Optimizer and Loss
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    #criterion = nn.CrossEntropyLoss()
    criterion = DiceLossIgnoringIndex0()

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # TensorBoard Setup
    writer = SummaryWriter(log_dir=log_dir)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.cuda(), masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.cuda(), masks.cuda()

                # Forward pass
                outputs = model(images)  # Outputs are at wanted resolution

                # Upsample the outputs to match the original mask resolution
                outputs_upsampled = torch.nn.functional.interpolate(outputs, size=(1024, 1024), mode='bilinear', align_corners=False)

                # Compute Loss
                loss = criterion(outputs_upsampled, masks)
                val_loss += loss.item()

                # Compute IoU
                #val_iou += calculate_iou(outputs_upsampled, masks, num_classes)
                val_iou += calculate_iou_ignore_index_0(outputs_upsampled, masks, num_classes)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        # Logging Metrics
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("IoU/val", avg_val_iou, epoch)

        print(f"Validation - Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}")

        # Step Scheduler
        scheduler.step(avg_val_loss)

        # Save Model Checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

    writer.close()

if __name__ == "__main__":
    # Example: Train with a preferred resolution of 512x512
    train(preferred_resolution=(720, 720))


