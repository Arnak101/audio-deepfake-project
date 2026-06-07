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

from model_ssl_linear import SSLLinearClassifier


PROJECT_DIR = Path("/home3/aaovsepian/tiny_rawnet")
DATA_ROOT = Path("/s3_ml_data/ahovsepyan/AT-ADD-Track1")
MUSAN_ROOT = Path("/s3_ml_data/ahovsepyan/musan")

TRAIN_AUDIO_DIR = DATA_ROOT / "train"
VAL_AUDIO_DIR = DATA_ROOT / "dev"
LABEL_DIR = DATA_ROOT / "label"

TRAIN_LABEL_CSV = LABEL_DIR / "train.csv"
VAL_LABEL_CSV = LABEL_DIR / "dev.csv"

OUTPUT_MODEL_PATH = PROJECT_DIR / "best_model_ssl_linear_musan.pt"
CHECKPOINT_PATH = PROJECT_DIR / "last_checkpoint_ssl_linear_musan.pt"

TARGET_SR = 16000
TARGET_LEN = 32000   # 2 sec

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8
NUM_WORKERS = 4
MAX_EPOCHS = 15
LR = 1e-5
CHECKPOINT_EVERY = 500

SSL_NAME = "wav2vec2_xlsr_300m"
UNFREEZE_LAST_N_LAYERS = 2

# -------------------------
# Augmentation config
# -------------------------
AUG_GAIN_PROB = 0.8
AUG_GAIN_MIN = 0.7
AUG_GAIN_MAX = 1.3

AUG_POLARITY_FLIP_PROB = 0.10

AUG_TIME_SHIFT_PROB = 0.3
AUG_TIME_SHIFT_MAX_SAMPLES = 1600

AUG_LOWPASS_PROB = 0.25
AUG_HIGHPASS_PROB = 0.25
AUG_EQ_PROB = 0.35

AUG_SPEED_PERTURB_PROB = 0.30
AUG_SPEED_CHOICES = [0.9, 1.1]

AUG_MUSAN_NOISE_PROB = 0.45
AUG_MUSAN_SPEECH_PROB = 0.35

AUG_NOISE_SNR_MIN = 5.0
AUG_NOISE_SNR_MAX = 25.0

AUG_SPEECH_SNR_MIN = 0.0
AUG_SPEECH_SNR_MAX = 20.0


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


def apply_lowpass(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    cutoff = random.uniform(2500.0, 7000.0)
    return torchaudio.functional.lowpass_biquad(wav, sample_rate, cutoff)


def apply_highpass(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    cutoff = random.uniform(40.0, 1200.0)
    return torchaudio.functional.highpass_biquad(wav, sample_rate, cutoff)


def apply_random_eq(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    mode = random.choice(["bass", "treble", "peaking"])

    if mode == "bass":
        gain_db = random.uniform(-8.0, 8.0)
        cutoff = random.uniform(80.0, 400.0)
        q = random.uniform(0.5, 1.2)
        return torchaudio.functional.bass_biquad(
            wav, sample_rate=sample_rate, gain=gain_db, central_freq=cutoff, Q=q
        )

    if mode == "treble":
        gain_db = random.uniform(-8.0, 8.0)
        cutoff = random.uniform(2500.0, 6000.0)
        q = random.uniform(0.5, 1.2)
        return torchaudio.functional.treble_biquad(
            wav, sample_rate=sample_rate, gain=gain_db, central_freq=cutoff, Q=q
        )

    gain_db = random.uniform(-8.0, 8.0)
    center_freq = random.uniform(300.0, 5000.0)
    q = random.uniform(0.4, 1.5)
    return torchaudio.functional.equalizer_biquad(
        wav, sample_rate=sample_rate, center_freq=center_freq, gain=gain_db, Q=q
    )


def apply_speed_perturb(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Manual speed perturbation without sox_effects.

    speed < 1.0  => slower
    speed > 1.0  => faster

    Implemented by:
    1) resample waveform to a virtual sample rate = sample_rate / speed
    2) interpret it back at sample_rate

    This changes duration/content speed and pitch together, which is fine
    for augmentation here.
    """
    speed = random.choice(AUG_SPEED_CHOICES)

    virtual_sr = max(1000, int(round(sample_rate / speed)))
    wav2 = torchaudio.functional.resample(
        wav.unsqueeze(0), sample_rate, virtual_sr
    ).squeeze(0)

    return wav2


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(torch.mean(x ** 2), min=1e-8))


def mix_with_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    clean_rms = rms(clean)
    noise_rms = rms(noise)

    desired_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scale = desired_noise_rms / torch.clamp(noise_rms, min=1e-8)

    mixed = clean + noise * scale
    return torch.clamp(mixed, -1.0, 1.0)


def load_audio_file(audio_path: Path, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = sf.read(str(audio_path), dtype="float32")

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = torch.from_numpy(wav)

    if sr != target_sr:
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), sr, target_sr
        ).squeeze(0)

    return wav


def repeat_or_crop_to_match(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    if wav.shape[0] == target_len:
        return wav
    if wav.shape[0] > target_len:
        start = random.randint(0, wav.shape[0] - target_len)
        return wav[start:start + target_len]
    repeats = math.ceil(target_len / wav.shape[0])
    return wav.repeat(repeats)[:target_len]


class ATADDDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        audio_dir: Path,
        musan_root: Path,
        target_sr=16000,
        target_len=32000,
        crop_train=True,
        apply_aug=False,
    ):
        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir)
        self.musan_root = Path(musan_root)
        self.target_sr = target_sr
        self.target_len = target_len
        self.crop_train = crop_train
        self.apply_aug = apply_aug

        self.items = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["name"]
                label = row["label"].strip().lower()
                audio_path = self.audio_dir / name
                if audio_path.exists():
                    self.items.append({"name": name, "label": label})

        self.musan_noise_files = sorted([p for p in (self.musan_root / "noise").rglob("*.wav")])
        self.musan_speech_files = sorted([p for p in (self.musan_root / "speech").rglob("*.wav")])

        print(f"{self.audio_dir}: found {len(self.items)} usable files from {self.csv_path}")
        print(f"MUSAN noise files: {len(self.musan_noise_files)}")
        print(f"MUSAN speech files: {len(self.musan_speech_files)}")

    def __len__(self):
        return len(self.items)

    def _mix_random_musan(self, clean: torch.Tensor, file_list, snr_min: float, snr_max: float) -> torch.Tensor:
        if len(file_list) == 0:
            return clean
        noise_path = random.choice(file_list)
        noise = load_audio_file(noise_path, self.target_sr)
        noise = repeat_or_crop_to_match(noise, clean.shape[0])
        snr_db = random.uniform(snr_min, snr_max)
        return mix_with_snr(clean, noise, snr_db)

    def _apply_augmentations(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() < AUG_GAIN_PROB:
            wav = wav * random.uniform(AUG_GAIN_MIN, AUG_GAIN_MAX)

        if random.random() < AUG_POLARITY_FLIP_PROB:
            wav = -wav

        if random.random() < AUG_TIME_SHIFT_PROB:
            shift = random.randint(-AUG_TIME_SHIFT_MAX_SAMPLES, AUG_TIME_SHIFT_MAX_SAMPLES)
            if shift != 0:
                wav = torch.roll(wav, shifts=shift, dims=0)

        if random.random() < AUG_LOWPASS_PROB:
            wav = apply_lowpass(wav, self.target_sr)

        if random.random() < AUG_HIGHPASS_PROB:
            wav = apply_highpass(wav, self.target_sr)

        if random.random() < AUG_EQ_PROB:
            wav = apply_random_eq(wav, self.target_sr)

        if random.random() < AUG_SPEED_PERTURB_PROB:
            wav = apply_speed_perturb(wav, self.target_sr)
            wav = random_or_center_crop(wav, self.target_len, train=True)

        if random.random() < AUG_MUSAN_NOISE_PROB:
            wav = self._mix_random_musan(
                wav,
                self.musan_noise_files,
                AUG_NOISE_SNR_MIN,
                AUG_NOISE_SNR_MAX,
            )

        if random.random() < AUG_MUSAN_SPEECH_PROB:
            wav = self._mix_random_musan(
                wav,
                self.musan_speech_files,
                AUG_SPEECH_SNR_MIN,
                AUG_SPEECH_SNR_MAX,
            )

        return torch.clamp(wav, -1.0, 1.0)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_path = self.audio_dir / item["name"]

        wav = load_audio_file(audio_path, self.target_sr)
        wav = random_or_center_crop(wav, self.target_len, train=self.crop_train)

        if self.apply_aug:
            wav = self._apply_augmentations(wav)

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
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
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

    return {
        "loss": total_loss / len(loader),
        "accuracy": acc,
        "f1_real": f1_real,
        "f1_fake": f1_fake,
        "macro_f1": macro_f1,
        "precision_real": precision_real,
        "recall_real": recall_real,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "confusion_matrix": cm,
    }


def print_metrics(title: str, metrics: dict):
    print(f"{title}:")
    print(f"  loss            : {metrics['loss']:.4f}")
    print(f"  accuracy        : {metrics['accuracy']:.4f}")
    print(f"  f1_real         : {metrics['f1_real']:.4f}")
    print(f"  f1_fake         : {metrics['f1_fake']:.4f}")
    print(f"  macro_f1        : {metrics['macro_f1']:.4f}")
    print(f"  precision_real  : {metrics['precision_real']:.4f}")
    print(f"  recall_real     : {metrics['recall_real']:.4f}")
    print(f"  precision_fake  : {metrics['precision_fake']:.4f}")
    print(f"  recall_fake     : {metrics['recall_fake']:.4f}")
    print("  confusion_matrix:")
    print(metrics["confusion_matrix"])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("TRAIN_LABEL_CSV:", TRAIN_LABEL_CSV)
    print("VAL_LABEL_CSV:", VAL_LABEL_CSV)
    print("TRAIN_AUDIO_DIR:", TRAIN_AUDIO_DIR)
    print("VAL_AUDIO_DIR:", VAL_AUDIO_DIR)
    print("MUSAN_ROOT:", MUSAN_ROOT)
    print("SSL_NAME:", SSL_NAME)
    print("UNFREEZE_LAST_N_LAYERS:", UNFREEZE_LAST_N_LAYERS)
    print("TARGET_SR:", TARGET_SR)
    print("TARGET_LEN:", TARGET_LEN)
    print("TRAIN_BATCH_SIZE:", TRAIN_BATCH_SIZE)
    print("VAL_BATCH_SIZE:", VAL_BATCH_SIZE)
    print("NUM_WORKERS:", NUM_WORKERS)
    print("LR:", LR)
    print("MAX_EPOCHS:", MAX_EPOCHS)

    train_ds = ATADDDataset(
        csv_path=TRAIN_LABEL_CSV,
        audio_dir=TRAIN_AUDIO_DIR,
        musan_root=MUSAN_ROOT,
        target_sr=TARGET_SR,
        target_len=TARGET_LEN,
        crop_train=True,
        apply_aug=True,
    )

    train_labels = [0 if item["label"] == "real" else 1 for item in train_ds.items]
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

    train_sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    val_clean_ds = ATADDDataset(
        csv_path=VAL_LABEL_CSV,
        audio_dir=VAL_AUDIO_DIR,
        musan_root=MUSAN_ROOT,
        target_sr=TARGET_SR,
        target_len=TARGET_LEN,
        crop_train=False,
        apply_aug=False,
    )

    val_aug_ds = ATADDDataset(
        csv_path=VAL_LABEL_CSV,
        audio_dir=VAL_AUDIO_DIR,
        musan_root=MUSAN_ROOT,
        target_sr=TARGET_SR,
        target_len=TARGET_LEN,
        crop_train=False,
        apply_aug=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_clean_loader = DataLoader(
        val_clean_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_aug_loader = DataLoader(
        val_aug_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = SSLLinearClassifier(
        ssl_name=SSL_NAME,
        unfreeze_last_n_layers=UNFREEZE_LAST_N_LAYERS,
        dropout=0.2,
    ).to(device)

    pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

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
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
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
                print(f"saved checkpoint at global_step {global_step} to: {CHECKPOINT_PATH}", flush=True)

        avg_train_loss = total_train_loss / len(train_loader)

        clean_metrics = evaluate(model, val_clean_loader, device, criterion)
        aug_metrics = evaluate(model, val_aug_loader, device, criterion)

        print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")
        print(f"train_loss: {avg_train_loss:.4f}")
        print_metrics("clean_val", clean_metrics)
        print_metrics("aug_val", aug_metrics)
        print("")

        if clean_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = clean_metrics["macro_f1"]
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

    print("Best clean val_macro_f1:", best_val_macro_f1)
    print("Training finished")


if __name__ == "__main__":
    main()