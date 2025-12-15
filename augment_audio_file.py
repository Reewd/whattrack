#!/usr/bin/env python3
"""
Audio augmentation script that applies the same augmentations from main.py to a single audio file.
Usage: python augment_audio_file.py <audio_file_path> [--aug-dir <path_to_aug_directory>]
"""

import argparse
import os
from pathlib import Path
import torchaudio
from audio.augmentation import AudioAugmentations
from audio.augmentations import (
    BackgroundNoiseMixing,
    ImpulseResponseAugmentation,
    PitchJitterAugmentation,
    VolumeAugmentation,
    ReverbAugmentation,
    BandPassFilterAugmentation,
)
import torch


def augment_audio_file(audio_path: str, aug_dir: str = "dataset/aug") -> None:
    """
    Load an audio file, apply augmentations, and save the result.
    
    Args:
        audio_path: Path to the input audio file
        aug_dir: Path to the augmentation directory containing bg, ir, etc.
    """
    # Verify input file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load audio file
    print(f"Loading audio file: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample to 8000 Hz if necessary (to match training setup)
    target_sample_rate = 8000
    if sample_rate != target_sample_rate:
        print(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz")
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)

    # convert to mono if stereo
    if waveform.shape[0] > 1:
        print("Converting to mono")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Create augmentations (same as in main.py)
    print("Setting up augmentations...")
    augmentations = AudioAugmentations(
        enabled_augmentations=[
            BackgroundNoiseMixing(
                files_path=f"{aug_dir}/bg",
                train=True,
                snr_range=(0, 15),
                amp_range=(0.1, 1.4),
                sample_rate=target_sample_rate
            ),
            ImpulseResponseAugmentation(
                ir_path=f"{aug_dir}/ir",
                train=True,
                sample_rate=target_sample_rate
            ),
            PitchJitterAugmentation(
                steps_range=(-2, 2),
                sample_rate=target_sample_rate,
                train=True
            ),
            VolumeAugmentation(
                gain_range=(-4.5, 4.5),
                scale_range=(0.5, 1.5),
                clipping=True,
                sample_rate=target_sample_rate,
                train=True
            ),
            ReverbAugmentation(
                reverb_amount=(0, 30),
                room_scale=(30, 80),
                damping=(30, 70),
                wet_dry_mix=(20, 50),
                sample_rate=target_sample_rate,
                train=True
            ),
            #BandPassFilterAugmentation(
            #    lower_range=(300, 500),
            #    upper_range=(3000, 3999),
            #    filter_order=4,
            #    sample_rate=target_sample_rate,
            #    train=True,
            #),
        ]
    )
    
    # Apply augmentations
    print("Applying augmentations...")
    augmented_waveform = augmentations.apply(waveform)
    
    # Generate output filename
    input_path = Path(audio_path)
    stem = input_path.stem
    suffix = input_path.suffix
    output_path = input_path.parent / f"{stem}_aug{suffix}"
    
    # Save augmented audio
    print(f"Saving augmented audio to: {output_path}")
    torchaudio.save(str(output_path), augmented_waveform, target_sample_rate)
    
    print(f"✓ Successfully augmented audio file!")
    print(f"  Input:  {audio_path}")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply audio augmentations to a single audio file"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file to augment"
    )
    parser.add_argument(
        "--aug-dir",
        type=str,
        default="dataset/aug",
        help="Path to augmentation directory (default: dataset/aug)"
    )
    
    args = parser.parse_args()
    
    augment_audio_file(args.audio_file, args.aug_dir)


if __name__ == "__main__":
    main()
