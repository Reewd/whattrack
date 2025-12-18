#!/bin/bash
# Convert all MP3 files to 8kHz mono WAV and remove corrupt files

set -e

# Check if source directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <source_directory> [num_jobs]"
    echo "Example: $0 fma_medium 16"
    exit 1
fi

SOURCE_DIR="$1"
NUM_JOBS="${2:-16}"  # Default to 16 if not provided
TEMP_LOG="/tmp/fma_convert_errors.log"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Directory $SOURCE_DIR does not exist"
    exit 1
fi

echo "Converting MP3s to WAV in $SOURCE_DIR using $NUM_JOBS parallel jobs..."
echo "Errors will be logged to $TEMP_LOG"

# Clear previous error log
> "$TEMP_LOG"

# Function to convert a single file
convert_file() {
    mp3_file="$1"
    wav_file="${mp3_file%.mp3}.wav"
    
    # Try to convert
    if ffmpeg -i "$mp3_file" -ar 8000 -ac 1 -y "$wav_file" -v error 2>&1; then
        # Verify the WAV file is valid
        if ffmpeg -v error -i "$wav_file" -f null - 2>&1; then
            # Success - remove original MP3
            rm "$mp3_file"
            echo "✓ $mp3_file"
        else
            # WAV is corrupt - remove both
            rm "$wav_file" 2>/dev/null || true
            rm "$mp3_file"
            echo "✗ CORRUPT: $mp3_file" >> "$TEMP_LOG"
            echo "✗ $mp3_file (corrupt)"
        fi
    else
        # Conversion failed - remove MP3
        rm "$mp3_file"
        rm "$wav_file" 2>/dev/null || true
        echo "✗ FAILED: $mp3_file" >> "$TEMP_LOG"
        echo "✗ $mp3_file (failed)"
    fi
}

export -f convert_file
export TEMP_LOG

# Find all MP3 files and convert in parallel
find "$SOURCE_DIR" -name "*.mp3" -type f -print0 | \
    xargs -0 -P "$NUM_JOBS" -I {} bash -c 'convert_file "$@"' _ {}

echo ""
echo "Conversion complete!"

if [ -s "$TEMP_LOG" ]; then
    echo ""
    echo "Errors found (see $TEMP_LOG):"
    wc -l < "$TEMP_LOG"
else
    echo "No errors encountered."
fi

# Count results
total_wav=$(find "$SOURCE_DIR" -name "*.wav" -type f | wc -l)
remaining_mp3=$(find "$SOURCE_DIR" -name "*.mp3" -type f | wc -l)

echo ""
echo "Results:"
echo "  WAV files: $total_wav"
echo "  Remaining MP3s: $remaining_mp3"
