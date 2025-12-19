import argparse
import sys
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
import torch
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm

# 1. Silence warnings (UserWarning, DeprecationWarning)
warnings.filterwarnings("ignore")

def split_file(args):
    """
    Worker function: Loads file, splits into 30s chunks, saves new files.
    """
    input_path, output_dir, chunk_duration_sec = args
    
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(input_path)

        # Enforce 8kHz first
        if sample_rate != 8000:
            waveform = F.resample(waveform, sample_rate, 8000)
            sample_rate = 8000

        # Calculate exact samples per chunk
        # 30s * 8000Hz = 240,000 samples
        samples_per_chunk = int(chunk_duration_sec * sample_rate)
        total_samples = waveform.size(1)

        generated_files = 0
        
        # Slicing Loop
        for i in range(0, total_samples, samples_per_chunk):
            # Calculate end index
            end = i + samples_per_chunk
            
            # Slice the tensor (Handles the last chunk automatically via min logic or python slicing)
            chunk_waveform = waveform[:, i:end]
            
            # Construct new filename: "original_name_part001.wav"
            # .stem removes extension, zfill adds padding (001, 002)
            part_name = f"{input_path.stem}_part{generated_files:03d}.wav"
            out_path = output_dir / part_name
            
            # Save
            torchaudio.save(out_path, chunk_waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)
            generated_files += 1

        return ("success", generated_files)

    except Exception as e:
        return ("error", f"{input_path.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Split audio files into precise 30s chunks.")
    parser.add_argument("input_folder", type=str, help="Folder containing source audio files")
    parser.add_argument("output_folder", type=str, help="Folder to save split chunks")
    parser.add_argument("--seconds", type=int, default=30, help="Duration of chunks in seconds (default: 30)")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of CPU workers")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_folder)
    output_dir = Path(args.output_folder)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    extensions = {".wav", ".mp3", ".flac"}
    files = [f for f in input_dir.rglob("*") if f.suffix.lower() in extensions]
    
    if not files:
        print("No audio files found.")
        sys.exit(0)

    print(f"Splitting {len(files)} files into {args.seconds}s chunks...")
    print(f"Target: {args.seconds}s @ 8000Hz = {args.seconds * 8000} samples")

    # Prepare arguments for parallel workers
    tasks = [(f, output_dir, args.seconds) for f in files]
    
    results = {"files_processed": 0, "chunks_created": 0, "errors": []}

    # Run Parallel
    with Pool(args.workers) as pool:
        for status, payload in tqdm(pool.imap_unordered(split_file, tasks), total=len(files)):
            if status == "success":
                results["files_processed"] += 1
                results["chunks_created"] += payload
            else:
                results["errors"].append(payload)

    # Summary
    print("\n" + "="*30)
    print("SPLIT COMPLETE")
    print("="*30)
    print(f"✅ Input Files:    {results['files_processed']}")
    print(f"📦 Total Chunks:   {results['chunks_created']}")
    print(f"❌ Errors:         {len(results['errors'])}")
    
    if results["errors"]:
        print("\nErrors:")
        for err in results["errors"][:5]:
            print(f"  - {err}")

if __name__ == "__main__":
    main()
