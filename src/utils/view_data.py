import argparse
import numpy as np

def view_data(npy_path, num_frames=5):
    """View basic info and sample frames from a .npy trajectory file."""
    data = np.load(npy_path)

    print("=" * 60)
    print(f"File:    {npy_path}")
    print(f"Shape:   {data.shape}")
    print(f"Dtype:   {data.dtype}")
    print(f"Min:     {data.min():.6f}")
    print(f"Max:     {data.max():.6f}")
    print(f"Mean:    {data.mean():.6f}")
    print(f"Std:     {data.std():.6f}")
    print("=" * 60)

    # Show first few frames
    num_frames = min(num_frames, data.shape[0])
    print(f"\nFirst {num_frames} frames:")
    for i in range(num_frames):
        print(f"\n  Frame {i}:")
        print(f"    {data[i]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View .npy trajectory data.")
    parser.add_argument("file", type=str, help="Path to .npy file")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of sample frames to display (default: 5)")
    args = parser.parse_args()

    view_data(args.file, num_frames=args.num_frames)
