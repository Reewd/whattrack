"""
Validate WAV files in FMA dataset using torchaudio and optionally remove invalid ones.
"""

import argparse
import torchaudio
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def validate_audio_file(file_path):
    """
    Validate a single audio file using torchaudio.
    Returns (file_path, is_valid, error_msg)
    """
    try:
        # Try to load the file
        waveform, sr = torchaudio.load(str(file_path))
        
        # Check if waveform is valid
        if waveform.shape[0] == 0 or waveform.shape[1] == 0:
            return (file_path, False, "Empty waveform")
        
        # Check sample rate
        if sr != 8000:
            return (file_path, False, f"Wrong sample rate: {sr}")
        
        return (file_path, True, None)
    
    except Exception as e:
        return (file_path, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Validate WAV files in FMA dataset using torchaudio"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory (e.g., fma_small, fma_medium, fma_large)"
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Actually remove invalid files (default: dry-run only)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help=f"Number of worker processes (default: {cpu_count()})"
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.dataset_path)
    
    if not source_dir.exists():
        print(f"Error: {source_dir} does not exist")
        return
    
    # Find all WAV files
    print(f"Finding WAV files in {source_dir}...")
    wav_files = list(source_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")
    
    if len(wav_files) == 0:
        print("No WAV files found. Convert MP3s to WAV first.")
        return
    
    # Validate in parallel
    print(f"Validating with {args.workers} workers...")
    
    invalid_files = []
    
    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(validate_audio_file, wav_files),
            total=len(wav_files),
            desc="Validating"
        ))
    
    # Process results
    for file_path, is_valid, error_msg in results:
        if not is_valid:
            invalid_files.append((file_path, error_msg))
            
            if args.remove:
                # Remove invalid file
                try:
                    file_path.unlink()
                    print(f"✗ Removed: {file_path} ({error_msg})")
                except Exception as e:
                    print(f"✗ Failed to remove {file_path}: {e}")
            else:
                print(f"✗ Would remove: {file_path} ({error_msg})")
    
    # Summary
    print("\n" + "="*60)
    print(f"Validation complete!")
    print(f"  Valid files: {len(wav_files) - len(invalid_files)}")
    print(f"  Invalid files: {len(invalid_files)}")
    
    if not args.remove and invalid_files:
        print(f"\n[DRY RUN] Would remove {len(invalid_files)} files")
        print("Run with --remove to actually delete them")
    elif args.remove and invalid_files:
        print(f"\nRemoved {len(invalid_files)} invalid files")
    
    if invalid_files and len(invalid_files) <= 50:
        print(f"\nInvalid files:")
        for file_path, error_msg in invalid_files:
            print(f"  {file_path}: {error_msg}")
    elif invalid_files:
        print(f"\nShowing first 50 invalid files:")
        for file_path, error_msg in invalid_files[:50]:
            print(f"  {file_path}: {error_msg}")
        print(f"  ... and {len(invalid_files) - 50} more")


if __name__ == "__main__":
    main()
