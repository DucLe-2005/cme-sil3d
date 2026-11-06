import os
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import logging
import wandb


###########################################
# Configuration and Logger Setup
###########################################
config = {
    "root_dir": os.path.expanduser("~/synthetic_cme_dataset/synthetic_coronagraphs/all_finished_cmes_lon10"),
    "max_cases": 15,  # Maximum number of case directories to use.
    "instrument": "c2",  # "c2", "c3", or "both"; one model per run.
    "target_size": (192, 192),
    "batch_size": 40,
    "num_workers": 8,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "checkpoint_path": "best_unet_baseline.pth",
    "alpha_loss": 0.5,
    "seed": 42,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###########################################
# Dataset Definition
###########################################
def collect_samples(events, instrument="both"):
    """Collect only the requested instrument, preserving case/file order."""
    if instrument not in ("c2", "c3", "both"):
        raise ValueError("instrument must be 'c2', 'c3', or 'both'")
    instruments = ("c2", "c3") if instrument == "both" else (instrument,)
    samples = []
    for event in events:
        for path in sorted(event.rglob("*.npz")):
            for name in instruments:
                samples.append((str(path), name))
    return samples


def load_raw_sample(file_path, instrument):
    """Load raw brightness, binary mask, and valid observation pixels."""
    if instrument not in ("c2", "c3"):
        raise ValueError("instrument must be 'c2' or 'c3'")
    with np.load(file_path, allow_pickle=False) as data:
        img = data[
            f"{instrument}_delta_total_brightness_msb"
        ].astype(np.float32)

        mask = data[
            f"{instrument}_cme_extent_bitmap"
        ].astype(np.float32)

        coords = data[f"{instrument}_coord_rsun"]

    # Calculate distance from the Sun's center
    x, y = np.meshgrid(coords, coords)
    radius = np.hypot(x, y)

    # C2 and C3 have different observable regions
    if instrument == "c2":
        inner, outer = 2.0, 6.0
    else:
        inner, outer = 3.7, 30.0

    # Identify valid observation pixels
    valid = (
        np.isfinite(img)
        & (radius >= inner)
        & (radius <= outer)
    )

    # Convert to binary mask
    mask = (mask > 0).astype(np.float32)

    # Add channel dimension: (H, W) -> (1, H, W)
    img = np.expand_dims(img, axis=0)
    mask = np.expand_dims(mask, axis=0)
    valid = np.expand_dims(
        valid.astype(np.float32), axis=0
    )

    return img, mask, valid


def fit_zscore(samples):
    """Fit pixel-weighted population mean/std from valid training pixels."""
    accumulators = {}
    for file_path, instrument in samples:
        img, _, valid = load_raw_sample(file_path, instrument)
        values = img[valid.astype(bool)].astype(np.float64)
        if not values.size:
            continue

        count = values.size
        mean = float(values.mean())
        m2 = float(np.sum((values - mean) ** 2))
        previous_count, previous_mean, previous_m2 = accumulators.get(
            instrument, (0, 0.0, 0.0)
        )
        total_count = previous_count + count
        delta = mean - previous_mean
        # Merge pixel statistics without retaining the entire training set.
        combined_mean = previous_mean + delta * count / total_count
        combined_m2 = (
            previous_m2 + m2
            + delta ** 2 * previous_count * count / total_count
        )
        accumulators[instrument] = (total_count, combined_mean, combined_m2)

    stats = {}
    for instrument in {instrument for _, instrument in samples}:
        if instrument not in accumulators:
            raise ValueError(f"No finite valid training pixels for {instrument}")
        count, mean, m2 = accumulators[instrument]
        stats[instrument] = {"mean": mean, "std": float(np.sqrt(m2 / count))}
    return stats


def load_processed_sample(file_path, instrument, normalization_stats):
    """Apply fixed training-set z-score statistics without clipping."""
    img, mask, valid = load_raw_sample(file_path, instrument)
    stats = normalization_stats[instrument]
    mean, std = float(stats["mean"]), float(stats["std"])
    if not np.isfinite([mean, std]).all() or std < 0:
        raise ValueError(f"Invalid normalization statistics for {instrument}")
    normalized = np.zeros_like(img)
    observed = valid.astype(bool)
    # A constant training distribution uses scale 1 to avoid division by zero.
    scale = std if std > 0 else 1.0
    normalized[observed] = (img[observed].astype(np.float64) - mean) / scale
    return normalized, mask, valid


class ProcessedSegmentationDataset(Dataset):

    def __init__(self, samples, normalization_stats):
        self.samples = samples
        self.normalization_stats = normalization_stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, instrument = self.samples[idx]

        return tuple(
            torch.from_numpy(array)
            for array in load_processed_sample(file_path, instrument, self.normalization_stats)
        )

###########################################
# Model, Loss Functions, and Metrics
###########################################
def dice_loss(outputs, targets, valid, smooth=1e-6):
    outputs = torch.sigmoid(outputs)

    # Exclude invalid pixels
    outputs = outputs * valid
    targets = targets * valid

    intersection = (outputs * targets).sum(dim=(1, 2, 3))
    union = outputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    loss = 1 - ((2. * intersection + smooth) / (union + smooth))
    return loss.mean()


def combined_loss(outputs, targets, valid, smooth=1e-6, alpha=0.5):
    bce = nn.BCEWithLogitsLoss(reduction="none")(outputs, targets)

    # Calculate BCE using only valid pixels
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)

    dice = dice_loss(outputs, targets, valid, smooth)
    return alpha * bce + (1 - alpha) * dice


def iou_metric(outputs, labels, valid, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    preds = (outputs > threshold).float()

    # Exclude invalid pixels
    preds = preds * valid
    labels = labels * valid

    intersection = (preds * labels).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

###########################################
# DataLoader Setup
###########################################

###########################################
# Checkpoint Saving and Loading
###########################################
def save_checkpoint(model, optimizer, epoch, loss, path, instrument, normalization_stats):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'instrument': instrument,
        'normalization': 'training_valid_zscore_v1',
        'normalization_stats': normalization_stats,
    }
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved at epoch {epoch+1} with loss {loss:.4f}")

###########################################
# Updated Training and Validation Functions
###########################################
def train_one_epoch(model, dataloader, optimizer, device, alpha=0.5):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for inputs, masks, valid in dataloader:
        inputs, masks, valid = (inputs.to(device), masks.to(device), valid.to(device))

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = combined_loss(outputs, masks, valid, alpha=alpha)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_iou += iou_metric(outputs, masks, valid) * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)

    return epoch_loss, epoch_iou

def validate(model, dataloader, device, alpha=0.5):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for inputs, masks, valid in dataloader:
            inputs, masks, valid = (inputs.to(device), masks.to(device), valid.to(device))

            outputs = model(inputs)

            loss = combined_loss(outputs, masks, valid, alpha=alpha)

            running_loss += loss.item() * inputs.size(0)
            running_iou += iou_metric(outputs, masks, valid) * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)

    return epoch_loss, epoch_iou


###########################################
# Main Training Loop
###########################################
def visualize_predictions(model, dataloader, device, num_samples=5, threshold=0.5):
    model.eval()
    images = []

    with torch.no_grad():
        for inputs, masks, valid in dataloader:
            outputs = model(inputs.to(device))
            preds = (torch.sigmoid(outputs).cpu() > threshold).numpy()

            inputs = inputs.numpy()
            masks = masks.numpy()
            valid = valid.numpy()

            for i in range(inputs.shape[0]):
                if len(images) >= num_samples:
                    break

                ground_truth = (masks[i, 0] * valid[i, 0]).astype(np.uint8)
                prediction = (preds[i, 0] * valid[i, 0]).astype(np.uint8)
                image = inputs[i, 0]

                # Scale only for visualization
                vmin, vmax = np.percentile(image, [1, 99])

                if vmax > vmin:
                    image = np.clip((image - vmin) / (vmax - vmin), 0, 1)

                images.append(
                    wandb.Image(
                        image,
                        caption=f"Sample {len(images) + 1}",
                        masks={
                            "ground_truth": {
                                "mask_data": ground_truth,
                                "class_labels": {0: "Background", 1: "CME"}
                            },
                            "prediction": {
                                "mask_data": prediction,
                                "class_labels": {0: "Background", 1: "CME"}
                            }
                        }
                    )
                )

            if len(images) >= num_samples:
                break

    wandb.log({"predictions": images})


###########################################
# Training and Final Evaluation
###########################################
def train():
    """Run one training trial using defaults plus any sweep overrides."""
    default_config = config.copy()
    # The sweep agent overrides searched values in run.config.
    with wandb.init(project="cme-segmentation", config=default_config) as run:
        trial_config = dict(run.config)
        run.summary["normalization"] = "training_valid_zscore_v1"
        torch.manual_seed(trial_config["seed"])

        root_dir = Path(trial_config["root_dir"])

        # Get all CME event directories
        event_dirs = sorted([
            path for path in root_dir.iterdir()
            if path.is_dir()
        ])

        # Select the same random cases for every trial.
        max_cases = trial_config["max_cases"]
        event_dirs = sorted(random.Random(trial_config["seed"]).sample(event_dirs, max_cases))
        print(f"Selected simulated cases: {len(event_dirs)}")

        # Each top-level directory is one simulated case (e.g. b080_I1_AR1).
        # Split cases before collecting samples so every nested sequence and both
        # instruments from a case belong to only one dataset.
        if len(event_dirs) < 2:
            raise ValueError("At least two simulated case directories are required to split the dataset.")

        # Split simulated cases: 80% training, 20% validation.
        train_events, val_events = train_test_split(
            event_dirs,
            test_size=0.2,
            random_state=trial_config["seed"]
        )

        train_samples = collect_samples(train_events, trial_config["instrument"])
        val_samples = collect_samples(val_events, trial_config["instrument"])
        logging.info(f"Training instrument: {trial_config['instrument']}; normalization: training-set z-score")
        if not train_samples or not val_samples:
            raise ValueError("Both training and validation cases must contain .npz samples.")

        # Fit only on training cases; validation reuses the same instrument statistics.
        logging.info("Computing training-set z-score statistics...")
        normalization_stats = fit_zscore(train_samples)
        run.summary["normalization_stats"] = normalization_stats
        logging.info(f"Normalization statistics: {normalization_stats}")

        # Create datasets
        train_dataset = ProcessedSegmentationDataset(train_samples, normalization_stats)
        val_dataset = ProcessedSegmentationDataset(val_samples, normalization_stats)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=trial_config["batch_size"],
            shuffle=True,
            num_workers=trial_config["num_workers"]
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=trial_config["batch_size"],
            shuffle=False,
            num_workers=trial_config["num_workers"]
        )

        print(f"Training events: {len(train_events)}")
        print(f"Validation events: {len(val_events)}")

        print(f"Training images: {len(train_dataset)}")
        print(f"Validation images: {len(val_dataset)}")

        # Record the case assignments so the absence of case overlap is reviewable.
        run.summary["train_cases"] = [path.name for path in train_events]
        run.summary["val_cases"] = [path.name for path in val_events]

        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation=None
        )
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        logging.info(f"Training on device: {device}")
        model = model.to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=trial_config["learning_rate"],
        )
        # Keep checkpoints outside the W&B run directory so they stay local.
        checkpoint_dir = Path(trial_config["checkpoint_path"]).parent / "checkpoints" / run.id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / Path(trial_config["checkpoint_path"]).name
        (checkpoint_dir / "normalization_stats.json").write_text(
            json.dumps(normalization_stats, indent=2) + "\n"
        )
        best_val_iou = float("-inf")
        for epoch in range(trial_config["num_epochs"]):
            train_loss, train_iou = train_one_epoch(
                model, train_loader, optimizer, device, alpha=trial_config["alpha_loss"]
            )
            val_loss, val_iou = validate(
                model, val_loader, device, alpha=trial_config["alpha_loss"]
            )
            if not np.isfinite(val_iou) or not np.isfinite(val_loss):
                raise RuntimeError("Validation metrics are non-finite; trial cannot be scored.")
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                save_checkpoint(
                    model, optimizer, epoch, val_loss, checkpoint_path,
                    trial_config["instrument"], normalization_stats,
                )
                run.summary["best_epoch"] = epoch + 1
                run.summary["best_val_loss"] = val_loss
            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "best_val_iou": best_val_iou,
            })
            logging.info(
                f"Epoch {epoch+1}/{trial_config['num_epochs']}: "
                f"Train Loss = {train_loss:.4f}, Train IoU = {train_iou:.4f}, "
                f"Val Loss = {val_loss:.4f}, Val IoU = {val_iou:.4f}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        run.summary["best_val_iou"] = best_val_iou
        run.summary["checkpoint_path"] = str(checkpoint_path)
        visualize_predictions(model, val_loader, device)




if __name__ == "__main__":
    # Normal run: use the defaults at the top of this file.
    # train()

    # Sweep: comment out train() above and uncomment this block.
    import yaml
    with Path(__file__).with_name("sweep.yaml").open() as sweep_file:
        sweep_config = yaml.safe_load(sweep_file)
    sweep_id = wandb.sweep(sweep_config, project="cme-segmentation")
    wandb.agent(sweep_id, function=train)
