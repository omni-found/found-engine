#!/usr/bin/env python3

import argparse
import datetime as dt.
import sys
from pathlib import Path

# import Elia's fetcher
from src.bench.fetch import get_jakel

def parse_args():
    parser = argparse.ArgumentParser(description='OmniBenchmark module')

    # Required by OmniBenchmark
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--name', type=str, required=True,
                       help='Module name/identifier')
    parser.add_argument('--dataset_name', type=str, help='Input file')
    return parser.parse_args()

def main():
    args = parse_args()

    # logging
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "dataset_name"):
        print(f"  {k}: {getattr(args, k)}")

    # make directory if doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running the fetch.")
    output_h5ad = output_dir / f"{args.name}_rawdata.h5ad"
    print(f"Output file will be: {output_h5ad}")

    cmd = f"get_{args.dataset_name}()"
    print(f"Running the fetch command: {cmd}")
    ad = eval(cmd)
    n_cells, n_features = ad.shape
    print(f"Got an AnnData with {n_cells} cells and {n_features} features.")
    print(f"Writing {output_h5ad}.")
    ad.write_h5ad(output_h5ad)

    print("Checking output.")
    stat = Path(output_h5ad).stat()  # raises if file missing
    print("Size:", stat.st_size, "bytes")
    print("Created:", dt.datetime.fromtimestamp(stat.st_ctime))

if __name__ == "__main__":
    main()


