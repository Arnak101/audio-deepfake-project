import csv
import math
import zipfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from model_mag_phase_mamba import MagPhaseMambaClassifier


PROJECT_DIR = Path("/home3/aaovsepian/tiny_rawnet")
DATA_ROOT = Path("/s3_ml_data/ahovsepyan/AT-ADD-Track1")

PROGRESS_AUDIO_DIR = DATA_ROOT / "eval_progress"
MODEL_PATH = PROJECT_DIR / "best_model_mag_phase_mamba.pt"

TARGET_SR = 16000
MODEL_INPUT_LEN = 32000   # 2 seconds
WINDOW_SEC = 2.0
STRIDE_SEC = 1.0

OUTPUT_CSV = PROJECT_DIR / "predict.csv"
OUTPUT_DEBUG_CSV = PROJECT_DIR / "predict_debug_mag_phase_mamba_multicrop_2s.csv"
OUTPUT_ZIP = PROJECT_DIR / "submission_mag_phase_mamba_multicrop_2s.zip"


def load_audio(audio_path: Path, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = sf.read(str(audio_path), dtype="float32")

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = torch.from_numpy(wav)

    if sr != target_sr:
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), sr, target_sr
        ).squeeze(0)

    return wav


def make_model_input(window: torch.Tensor, target_len: int = 32000) -> torch.Tensor:
    length = window.shape[0]

    if length == target_len:
        return window

    if length > target_len:
        return window[:target_len]

    repeats = math.ceil(target_len / length)
    return window.repeat(repeats)[:target_len]


def build_windows(wav: torch.Tensor, window_sec: float, stride_sec: float, sr: int = 16000):
    window_len = int(window_sec * sr)
    stride_len = int(stride_sec * sr)
    total_len = wav.shape[0]

    if total_len <= window_len:
        return [wav]

    windows = []
    start = 0

    while start + window_len <= total_len:
        windows.append(wav[start:start + window_len])
        start += stride_len

    last_start = total_len - window_len
    if len(windows) == 0 or last_start > (len(windows) - 1) * stride_len:
        windows.append(wav[last_start:last_start + window_len])

    return windows


def predict_file(model, audio_path: Path, device, window_sec: float, stride_sec: float):
    wav = load_audio(audio_path, target_sr=TARGET_SR)
    windows = build_windows(wav, window_sec=window_sec, stride_sec=stride_sec, sr=TARGET_SR)

    model_inputs = []
    for w in windows:
        x = make_model_input(w, target_len=MODEL_INPUT_LEN)
        model_inputs.append(x)

    batch = torch.stack(model_inputs, dim=0).to(device)

    with torch.no_grad():
        logits = model(batch)                  # [num_windows]
        probs = torch.sigmoid(logits)          # fake probability
        mean_prob_fake = probs.mean().item()
        max_prob_fake = probs.max().item()

    pred_label = "fake" if mean_prob_fake >= 0.5 else "real"
    return pred_label, mean_prob_fake, max_prob_fake, len(windows)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    print("PROGRESS_AUDIO_DIR:", PROGRESS_AUDIO_DIR)
    print("MODEL_PATH:", MODEL_PATH)
    print("WINDOW_SEC:", WINDOW_SEC)
    print("STRIDE_SEC:", STRIDE_SEC)

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

    state_dict = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded model weights successfully.")

    audio_files = sorted([p for p in PROGRESS_AUDIO_DIR.iterdir() if p.is_file()])
    print("num files found:", len(audio_files))

    submission_rows = []
    debug_rows = []

    for idx, audio_path in enumerate(audio_files):
        pred_label, mean_prob_fake, max_prob_fake, num_windows = predict_file(
            model=model,
            audio_path=audio_path,
            device=device,
            window_sec=WINDOW_SEC,
            stride_sec=STRIDE_SEC,
        )

        submission_rows.append([audio_path.name, pred_label])
        debug_rows.append([
            audio_path.name,
            pred_label,
            f"{mean_prob_fake:.6f}",
            f"{max_prob_fake:.6f}",
            num_windows,
        ])

        if (idx + 1) % 500 == 0:
            print(f"processed {idx + 1} files")

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "predict"])
        writer.writerows(submission_rows)

    print(f"Saved submission csv to: {OUTPUT_CSV}")

    with open(OUTPUT_DEBUG_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "predict", "mean_prob_fake", "max_prob_fake", "num_windows"])
        writer.writerows(debug_rows)

    print(f"Saved debug csv to: {OUTPUT_DEBUG_CSV}")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_CSV, arcname="predict.csv")

    print(f"Saved submission zip to: {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()