import os
import math
import random
from pathlib import Path

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from model_tiny_rawnet2 import TinyRawNet2


HF_DATASET_NAME = "xieyuankun/AT-ADD-Track1"

# Текущая структура папок у тебя именно такая:
TRAIN_AUDIO_DIR = Path("atadd/T1/train/train")
VAL_AUDIO_DIR = Path("atadd/T1/dev/dev")


def random_or_center_crop(wav: torch.Tensor, target_len: int = 16000, train: bool = True) -> torch.Tensor:
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
    def __init__(self, hf_split, audio_dir: Path, target_sr=16000, target_len=16000, train=True):
        self.ds = hf_split
        self.audio_dir = Path(audio_dir)
        self.target_sr = target_sr
        self.target_len = target_len
        self.train = train

        # оставляем только те элементы, для которых файл реально существует
        self.items = []
        for item in self.ds:
            audio_path = self.audio_dir / item["name"]
            if audio_path.exists():
                self.items.append(item)

        print(f"{self.audio_dir}: found {len(self.items)} usable files out of {len(self.ds)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_path = self.audio_dir / item["name"]

        wav, sr = torchaudio.load(str(audio_path))

        # stereo -> mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = wav.squeeze(0)

        # resample if needed
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.target_sr).squeeze(0)

        wav = random_or_center_crop(wav, self.target_len, train=self.train)

        # label: real/fake -> 0/1
        label_str = item["label"].strip().lower()
        if label_str == "real":
            label = 0
        else:
            label = 1

        return wav, torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)

    f1_real = f1_score(all_labels, all_preds, pos_label=0)
    f1_fake = f1_score(all_labels, all_preds, pos_label=1)
    macro_f1 = (f1_real + f1_fake) / 2.0

    cm = confusion_matrix(all_labels, all_preds)

    return total_loss, acc, f1_real, f1_fake, macro_f1, cm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = load_dataset(HF_DATASET_NAME)
    print(ds)

    train_split = ds["train"]
    val_split = ds["validation"]

    train_ds = ATADDDataset(
        train_split,
        audio_dir=TRAIN_AUDIO_DIR,
        target_sr=16000,
        target_len=16000,   # 1 second
        train=True,
    )

    val_ds = ATADDDataset(
        val_split,
        audio_dir=VAL_AUDIO_DIR,
        target_sr=16000,
        target_len=16000,
        train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = TinyRawNet2(
        sample_rate=16000,
        input_samples=16000,
        sinc_out=8,
        sinc_kernel=129,
        sinc_stride=4,
        channels_stage1=8,
        channels_stage2=12,
        gru_hidden=8,
        fc_hidden=12,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_macro_f1 = -1.0

    for epoch in range(10):
        model.train()
        total_train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        val_loss, val_acc, val_f1_real, val_f1_fake, val_macro_f1, val_cm = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/10 | "
            f"train_loss={total_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1_real={val_f1_real:.4f} | "
            f"val_f1_fake={val_f1_fake:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f}"
        )
        print("val_confusion_matrix:")
        print(val_cm)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), "best_model.pt")
            print("saved best_model.pt")

    print("Training finished")


if __name__ == "__main__":
    main()