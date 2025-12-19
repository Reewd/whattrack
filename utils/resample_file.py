import argparse
import sys
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import torch
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
def process_file(file_path):
    """
    Loads, resamples to 8k, trims to multiple of 2s, and saves.
    """
    try:
        # 1. Load Audio
        # waveform shape: [channels, time]
        waveform, sample_rate = torchaudio.load(file_path)

        # 2. Resample if necessary
        if sample_rate != 8000:
            waveform = F.resample(waveform, sample_rate, 8000)

        # 3. Calculate Valid Length
        # Each "chunk" is 2 seconds * 8000 Hz = 16000 samples
        chunk_size = 16000
        total_samples = waveform.size(1)
        
        # Integer division automatically drops the remainder
        # e.g., 35000 // 16000 = 2 chunks -> 2 * 16000 = 32000 samples
        valid_samples = (total_samples // chunk_size) * chunk_size
        
        # 4. Filter out short files
        if valid_samples == 0:
            #os.remove(file_path)
            return ("deleted_short", file_path.name)

        # 5. Trim (Slice the Tensor)
        # Only slice if we actually need to remove samples
        if valid_samples < total_samples:
            waveform = waveform[:, :valid_samples]

        # 6. Save (Overwrite)
        # encoding="PCM_S" ensures standard 16-bit WAV compatibility
        torchaudio.save(file_path, waveform, 8000, encoding="PCM_S", bits_per_sample=16)
        
        return ("success", file_path.name)

    except Exception as e:
        return ("error", f"{file_path.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Strictly enforce 8kHz and 2s-multiple duration using Torchaudio.")
    parser.add_argument("folder", type=str, help="Folder containing audio files")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of CPU workers")
    
    args = parser.parse_args()
    directory = Path(args.folder)
    
    if not directory.exists():
        print(f"Directory not found: {directory}")
        sys.exit(1)

    # Gather files
    files = list(directory.rglob("*.wav"))
    if not files:
        print("No .wav files found.")
        sys.exit(0)

    print(f"Processing {len(files)} files using {args.workers} workers...")

    results = {"success": 0, "deleted_short": 0, "errors": []}

    # Run in Parallel
    with Pool(args.workers) as pool:
        for status, msg in tqdm(pool.imap_unordered(process_file, files), total=len(files)):
            if status == "success":
                results["success"] += 1
            elif status == "deleted_short":
                results["deleted_short"] += 1
            elif status == "error":
                results["errors"].append(msg)

    # Summary
    print("\n" + "="*30)
    print("PROCESSING COMPLETE")
    print("="*30)
    print(f"✅ Converted/Trimmed: {results['success']}")
    print(f"🗑️ Deleted (Too short): {results['deleted_short']}")
    print(f"❌ Errors:              {len(results['errors'])}")
    
    if results["errors"]:
        print("\nError Logs:")
        for err in results["errors"][:10]:
            print(f"  - {err}")

if __name__ == "__main__":
    main()
