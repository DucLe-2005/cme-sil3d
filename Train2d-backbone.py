import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import logging
import wandb

###########################################
# Configuration and Logger Setup
###########################################
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_config.json")
config = {
    "csv_file": "/data/CME_reconstruction/cme_silhouette/processed_dataset_new/file_list.csv",
    "root_dir": "/data/CME_reconstruction/cme_silhouette/processed_dataset_new",
    "target_size": (832, 832),
    "batch_size": 12,
    "num_workers": 8,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "checkpoint_path": "best_model.pth",
    "model_name": "upernet",
    "alpha_loss": 0.5,
    "seed": 42,
    "visualization_samples": 30,
    "threshold": 0.5,
    "wandb": {
        "project": "cme-reconstruction",
        "entity": None,
        "run_name": None,
        "tags": ["segmentation"],
        "mode": "online",
    },
}

with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
    config.update(json.load(config_file))

# JSON can override target_size with a list; the dataset expects a tuple.
if isinstance(config["target_size"], list):
    config["target_size"] = tuple(config["target_size"])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

###########################################
# Utility Functions
###########################################
def pad_to_target(img, target_height=832, target_width=832):
    # Replace NaN values with 0
    img = np.nan_to_num(img, nan=0.0)
    if img.ndim == 2:
        h, w = img.shape
        pad_h = target_height - h
        pad_w = target_width - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = np.pad(img, ((top, bottom), (left, right)), mode='constant', constant_values=0)
    elif img.ndim == 3:
        # Assuming channel-first format (C, H, W)
        c, h, w = img.shape
        pad_h = target_height - h
        pad_w = target_width - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = np.pad(img, ((0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)
    else:
        raise ValueError("Unsupported image dimensions")
    return padded

def overlay_mask(image, mask, alpha=0.5):
    """Overlay predicted mask (binary) onto grayscale image."""
    # If image is 2D, convert to RGB for overlay
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image
    mask_rgb = np.zeros_like(image_rgb)
    # Mark the red channel in areas where mask==1
    mask_rgb[:, :, 0] = mask * 255
    overlay = image_rgb.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha
    return overlay.astype(np.uint8)

###########################################
# Dataset Definition
###########################################
class ProcessedSegmentationDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_size=(832,832), transform=None):
        self.file_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.file_df)

    def __getitem__(self, idx):
        rel_path = self.file_df.iloc[idx]["file_path"]
        file_path = os.path.join(self.root_dir, rel_path)
        sample = np.load(file_path, allow_pickle=True).item()
        
        # Process input image
        img = sample["input"]
        img = np.nan_to_num(img, nan=0.0)
        img = np.expand_dims(img, axis=0).astype(np.float32)  # shape (1, H, W)
        
        # Process mask (ensure it's binary)
        mask = sample["mask"]
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # shape (1, H, W)
        
        # Pad to target size
        img = pad_to_target(img, target_height=self.target_size[0], target_width=self.target_size[1])
        mask = pad_to_target(mask, target_height=self.target_size[0], target_width=self.target_size[1])
        
        if self.transform:
            img, mask = self.transform(img, mask)
            
        return torch.tensor(img), torch.tensor(mask)

###########################################
# Model, Loss Functions, and Metrics
###########################################
# Add/remove models here. Each model uses the same encoder and training code.
model_classes = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "upernet": smp.UPerNet,
    "segformer": smp.Segformer,
}

def build_model(model_class):
    return model_class(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None,
    )

def dice_loss(outputs, targets, smooth=1e-6):
    outputs = torch.sigmoid(outputs)
    intersection = (outputs * targets).sum(dim=(1,2,3))
    union = outputs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    loss = 1 - ((2. * intersection + smooth) / (union + smooth))
    return loss.mean()

def combined_loss(outputs, targets, smooth=1e-6, alpha=0.5):
    bce = nn.BCEWithLogitsLoss()(outputs, targets)
    dice = dice_loss(outputs, targets, smooth)
    return alpha * bce + (1 - alpha) * dice

def iou_metric(outputs, labels, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    preds = (outputs > threshold).float()
    intersection = (preds * labels).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + labels.sum(dim=(1,2,3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

###########################################
# DataLoader Setup
###########################################
dataset = ProcessedSegmentationDataset(
    csv_file=config["csv_file"], 
    root_dir=config["root_dir"], 
    target_size=config["target_size"]
)
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=config["seed"])
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

###########################################
# Checkpoint Saving and Loading
###########################################
def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved at epoch {epoch+1} with loss {loss:.4f}")

###########################################
# Updated Training and Validation Functions
###########################################
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    for inputs, masks in dataloader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = combined_loss(outputs, masks, alpha=config["alpha_loss"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_iou += iou_metric(outputs, masks) * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)
    return epoch_loss, epoch_iou

def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = combined_loss(outputs, masks, alpha=config["alpha_loss"])
            running_loss += loss.item() * inputs.size(0)
            running_iou += iou_metric(outputs, masks) * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)
    return epoch_loss, epoch_iou


###########################################
# Main Training Loop
###########################################
def visualize_predictions(model, dataloader, device, model_name, num_samples=30, threshold=0.5):
    model.eval()
    all_inputs, all_masks, all_outputs = [], [], []
    total_collected = 0

    # Loop over batches until we collect enough samples.
    for batch in dataloader:
        inputs, masks = batch
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        outputs = torch.sigmoid(outputs).cpu().numpy()
        inputs = inputs.cpu().numpy()
        masks = masks.cpu().numpy()

        all_inputs.append(inputs)
        all_masks.append(masks)
        all_outputs.append(outputs)

        total_collected += inputs.shape[0]
        if total_collected >= num_samples:
            break

    # Concatenate the batches along the first dimension.
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    # Use the minimum of num_samples and collected samples.
    num_samples = min(num_samples, all_inputs.shape[0])

    # Create subplots: 4 columns per sample (original, ground truth, predicted, overlay)
    fig, axs = plt.subplots(num_samples, 4, figsize=(25, 5*num_samples))
    for i in range(num_samples):
        # Original Input
        axs[i, 0].imshow(all_inputs[i, 0, :, :], cmap='gray')
        axs[i, 0].set_title("Original Image")
        axs[i, 0].axis("off")
        
        # Ground Truth Mask
        axs[i, 1].imshow(all_masks[i, 0, :, :], cmap='gray')
        axs[i, 1].set_title("Ground Truth Mask")
        axs[i, 1].axis("off")
        
        # Predicted Mask (binary)
        pred_mask = (all_outputs[i, 0, :, :] > threshold).astype(np.float32)
        axs[i, 2].imshow(pred_mask, cmap='gray')
        axs[i, 2].set_title("Predicted Mask")
        axs[i, 2].axis("off")
        
        # Overlay: Input + Predicted Mask (using imshow overlay technique)
        axs[i, 3].imshow(all_inputs[i, 0, :, :], cmap='gray')
        axs[i, 3].imshow(pred_mask, cmap='Reds', alpha=0.3)
        axs[i, 3].set_title("Overlay: Input + Prediction")
        axs[i, 3].axis("off")

    plt.tight_layout()
    prediction_path = f"predicted_samples_{model_name}.png"
    plt.savefig(prediction_path, dpi=150, bbox_inches="tight")
    plt.show()
    return prediction_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Training on device: {device}")
model_name = config["model_name"]
if model_name not in model_classes:
    raise ValueError(f"Unknown model_name '{model_name}'. Available: {', '.join(model_classes)}")

wandb_settings = config["wandb"]
run = wandb.init(
    project=wandb_settings["project"],
    entity=wandb_settings.get("entity"),
    name=wandb_settings.get("run_name") or model_name,
    tags=wandb_settings.get("tags"),
    mode=wandb_settings.get("mode", "online"),
    config={key: value for key, value in config.items() if key != "wandb"},
)
model = build_model(model_classes[model_name]).to(device)
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
train_losses, train_ious = [], []
val_losses, val_ious = [], []
best_loss = float('inf')
checkpoint_path = f"{model_name}_{config['checkpoint_path']}"

for epoch in range(config["num_epochs"]):
    train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_iou = validate(model, val_loader, device)

    train_losses.append(train_loss)
    train_ious.append(train_iou)
    val_losses.append(val_loss)
    val_ious.append(val_iou)

    logging.info(f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss = {train_loss:.4f}, Train IoU = {train_iou:.4f}, Val Loss = {val_loss:.4f}, Val IoU = {val_iou:.4f}")
    run.log({
        "epoch": epoch + 1,
        "train/loss": train_loss,
        "train/iou": train_iou,
        "validation/loss": val_loss,
        "validation/iou": val_iou,
    })

    if train_loss < best_loss:
        best_loss = train_loss
        save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)



prediction_path = visualize_predictions(
    model,
    val_loader,
    device,
    model_name,
    num_samples=config["visualization_samples"],
    threshold=config["threshold"],
)
run.log({
    "plots/validation_predictions": wandb.Image(prediction_path),
    "best_train_loss": best_loss,
})

model_artifact = wandb.Artifact(
    name=f"{model_name}-model",
    type="model",
    metadata={"model_name": model_name, "best_train_loss": best_loss},
)
model_artifact.add_file(checkpoint_path)
run.log_artifact(model_artifact)
run.finish()
