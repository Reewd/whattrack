import argparse
import torchaudio
from pathlib import Path
from tqdm import tqdm
import sys

def validate_audio_files(directory_path):
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)

    total_files = 0
    corrupted_files = []
    wrong_sr_files = []
    wrong_duration_files = []
    
    # 1. Find all .wav files recursively
    audio_files = list(directory.rglob("*.wav")) 
    
    if not audio_files:
        print(f"No .wav files found in '{directory}'.")
        sys.exit(0)

    print(f"Scanning {len(audio_files)} files in '{directory}'...\n")

    for file_path in tqdm(audio_files, unit="file"):
        total_files += 1
        
        try:
            # 2. Check header metadata (Fast)
            metadata = torchaudio.info(str(file_path))
            
            # 3. Check Sampling Rate (Must be 8000)
            if metadata.sample_rate != 8000:
                wrong_sr_files.append((file_path, metadata.sample_rate))
                continue

            # 4. Check Duration (Must be multiple of 16000 samples)
            if metadata.num_frames % 16000 != 0:
                wrong_duration_files.append((file_path, metadata.num_frames))
                continue
                
        except Exception as e:
            # 5. Check Corruption
            corrupted_files.append((file_path, str(e)))

    # --- REPORT ---
    print("\n" + "="*40)
    print("SCAN REPORT")
    print("="*40)
    
    valid_count = total_files - len(corrupted_files) - len(wrong_sr_files) - len(wrong_duration_files)
    
    print(f"Total files:          {total_files}")
    print(f"✅ Valid files:       {valid_count}")
    print(f"❌ Corrupted:         {len(corrupted_files)}")
    print(f"⚠️ Wrong SR (!= 8k):  {len(wrong_sr_files)}")
    print(f"⚠️ Bad Len (!= 2s*N): {len(wrong_duration_files)}")
    
    if corrupted_files:
        print("\n[Corrupted Files]")
        for f, err in corrupted_files: print(f"  rm \"{f}\"  # {err}")

    if wrong_sr_files:
        print("\n[Wrong Sample Rate]")
        for f, sr in wrong_sr_files: print(f"  rm \"{f}\"  # {sr}Hz")

    if wrong_duration_files:
        print("\n[Bad Duration]")
        for f, frames in wrong_duration_files: print(f"  rm \"{f}\"  # {frames} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan audio files for strict format compliance.")
    parser.add_argument("folder", type=str, help="Path to the folder containing .wav files")
    
    args = parser.parse_args()
    
    validate_audio_files(args.folder)
