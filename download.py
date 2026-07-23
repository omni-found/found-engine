#!/usr/bin/env python3

import argparse
import sys

# import Elia's fetcher
from src.bench.fetch import get_jakel

def parse_args():
    parser = argparse.ArgumentParser(description='OmniBenchmark module')

    # Required by OmniBenchmark
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--name', type=str, required=True,
                       help='Module name/identifier')
    parser.add_argument('--dataset', type=str, help='Input file')
    return parser.parse_args()

def main():
    args = parse_args()

    # logging
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "dataset"):
        print(f"  {k}: {getattr(args, k)}")

    # TODO: Implement your module logic
    # Process the data using main function
    # process_data(args)

if __name__ == "__main__":
    main()


