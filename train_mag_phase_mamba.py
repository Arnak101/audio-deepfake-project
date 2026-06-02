import csv
import math
import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from model_mag_phase_mamba import MagPhaseMambaClassifier


PROJECT_DIR = Path("/home3/aaovsepian/tiny_rawnet")
DATA_ROOT = Path("/s3_ml_data/ahovsepyan/AT-ADD-Track1")

TRAIN_AUDIO_DIR = DATA_ROOT / "train"
VAL_AUDIO_DIR = DATA_ROOT / "dev"
LABEL_DIR = DATA_ROOT / "label"

TRAIN_LABEL_CSV = LABEL_DIR / "train.csv"
VAL_LABEL_CSV = LABEL_DIR / "dev.csv"

OUTPUT_MODEL_PATH = PROJECT_DIR / "best_model_mag_phase_mamba.pt"
CHECKPOINT_PATH = PROJECT_DIR / "last_checkpoint_mag_phase_mamba.pt"

# =========================
TARGET_SR = 16000
TARGET_LEN = 32000   # 2 sec
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 32
NUM_WORKERS = 2
MAX_EPOCHS = 20
LR = 1e-4
CHECKPOINT_EVERY = 500
# =========================


def random_or_center_crop(wav: torch.Tensor, target_len: int = 32000, train: bool = True) -> torch.Tensor:
    length = wav.shape[0]

    if length == target_len:
        return wav

    if length > target_len:
        if train:
            start = random.randint(0, length - target_len)
        else:
            start = (length - target_len) // 2
        return wav[start:start + target_len]

    repeats = math.ceil(target_len / length)
    wav = wav.repeat(repeats)[:target_len]
    return wav


class ATADDDataset(Dataset):
    def __init__(self, csv_path: Path, audio_dir: Path, target_sr=16000, target_len=32000, train=True):
        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir)
        self.target_sr = target_sr
        self.target_len = target_len
        self.train = train

        self.items = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["name"]
                label = row["label"].strip().lower()
                audio_path = self.audio_dir / name

                if audio_path.exists():
                    self.items.append({
                        "name": name,
                        "label": label,
                    })

        print(f"{self.audio_dir}: found {len(self.items)} usable files from {self.csv_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_path = self.audio_dir / item["name"]

        wav, sr = sf.read(str(audio_path), dtype="float32")

        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        wav = torch.from_numpy(wav)

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, self.target_sr
            ).squeeze(0)

        wav = random_or_center_crop(wav, self.target_len, train=self.train)

        label = 0 if item["label"] == "real" else 1
        return wav, torch.tensor(label, dtype=torch.float32)


def evaluate(model, loader, device, criterion):
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # [B]
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.long().cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)

    f1_real = f1_score(all_labels, all_preds, pos_label=0, zero_division=0)
    f1_fake = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    macro_f1 = (f1_real + f1_fake) / 2.0

    precision_real = precision_score(all_labels, all_preds, pos_label=0, zero_division=0)
    recall_real = recall_score(all_labels, all_preds, pos_label=0, zero_division=0)

    precision_fake = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall_fake = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return (
        total_loss,
        acc,
        f1_real,
        f1_fake,
        macro_f1,
        precision_real,
        recall_real,
        precision_fake,
        recall_fake,
        cm,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    print("TRAIN_LABEL_CSV:", TRAIN_LABEL_CSV)
    print("VAL_LABEL_CSV:", VAL_LABEL_CSV)
    print("TRAIN_AUDIO_DIR:", TRAIN_AUDIO_DIR)
    print("VAL_AUDIO_DIR:", VAL_AUDIO_DIR)

    train_ds = ATADDDataset(
        csv_path=TRAIN_LABEL_CSV,
        audio_dir=TRAIN_AUDIO_DIR,
        target_sr=TARGET_SR,
        target_len=TARGET_LEN,
        train=True,
    )

    train_labels = []
    for item in train_ds.items:
        label = 0 if item["label"] == "real" else 1
        train_labels.append(label)

    class_count_real = sum(1 for x in train_labels if x == 0)
    class_count_fake = sum(1 for x in train_labels if x == 1)

    print("class_count_real:", class_count_real)
    print("class_count_fake:", class_count_fake)

    sample_weights = []
    for label in train_labels:
        if label == 0:
            sample_weights.append(1.0 / class_count_real)
        else:
            sample_weights.append(1.0 / class_count_fake)

    sample_weights = torch.DoubleTensor(sample_weights)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    val_ds = ATADDDataset(
        csv_path=VAL_LABEL_CSV,
        audio_dir=VAL_AUDIO_DIR,
        target_sr=TARGET_SR,
        target_len=TARGET_LEN,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = MagPhaseMambaClassifier(
        sample_rate=TARGET_SR,
        n_fft=512,
        hop_length=160,
        win_length=400,
        mag_out_dim=256,
        phase_model_dim=192,
        phase_blocks=4,
        phase_out_dim=256,
        dropout=0.2,
    ).to(device)

    # stronger weight on real class, like before
    pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0
    best_val_macro_f1 = -1.0
    global_step = 0

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_macro_f1 = checkpoint["best_val_macro_f1"]
        global_step = checkpoint.get("global_step", 0)
        print(f"Resuming from epoch {start_epoch}, global_step {global_step}")
    else:
        print("Starting training from scratch")

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # [B]
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            global_step += 1

            if batch_idx % 200 == 0:
                print(
                    f"epoch {epoch+1} | batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f}",
                    flush=True,
                )

            if global_step % CHECKPOINT_EVERY == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                }
                torch.save(checkpoint, str(CHECKPOINT_PATH))
                print(
                    f"saved checkpoint at global_step {global_step} to: {CHECKPOINT_PATH}",
                    flush=True,
                )

        (
            val_loss,
            val_acc,
            val_f1_real,
            val_f1_fake,
            val_macro_f1,
            val_precision_real,
            val_recall_real,
            val_precision_fake,
            val_recall_fake,
            val_cm,
        ) = evaluate(model, val_loader, device, criterion)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{MAX_EPOCHS} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1_real={val_f1_real:.4f} | "
            f"val_f1_fake={val_f1_fake:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f} | "
            f"val_precision_real={val_precision_real:.4f} | "
            f"val_recall_real={val_recall_real:.4f} | "
            f"val_precision_fake={val_precision_fake:.4f} | "
            f"val_recall_fake={val_recall_fake:.4f}"
        )
        print("val_confusion_matrix:")
        print(val_cm)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), str(OUTPUT_MODEL_PATH))
            print(f"saved best model to: {OUTPUT_MODEL_PATH}")

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
        }
        torch.save(checkpoint, str(CHECKPOINT_PATH))
        print(f"saved end-of-epoch checkpoint to: {CHECKPOINT_PATH}")

    print("Best val_macro_f1:", best_val_macro_f1)
    print("Training finished")


if __name__ == "__main__":
    main()