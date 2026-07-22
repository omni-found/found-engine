#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from main import process_data

def parse_args():
    parser = argparse.ArgumentParser(description='OmniBenchmark module')

    # Required by OmniBenchmark
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--name', type=str, required=True,
                       help='Module name/identifier')
    # Add your custom input arguments here
    # Example:
    # parser.add_argument('--input', type=str, help='Input file')

    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Output directory: {args.output_dir}")
    print(f"Module name: {args.name}")

    # TODO: Implement your module logic
    # Process the data using main function
    process_data(args)

if __name__ == "__main__":
    main()
