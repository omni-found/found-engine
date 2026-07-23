#!/usr/bin/env python3

import argparse
import datetime as dt
import sys
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser(description='OmniBenchmark module')

    # Required by OmniBenchmark
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--name', type=str, required=True,
                       help='Module name/identifier')
    parser.add_argument('--data_ad', type=str, help='Input dataset')
    parser.add_argument('--embed_method', type=str, help='Embedding method')
    parser.add_argument('--dim', type=int, help='Number of dimensions')
    return parser.parse_args()

def main():
    print("Parsing args.")
    args = parse_args()

    # logging
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "data_ad", "embed_method", "dim"):
        print(f"  {k}: {getattr(args, k)}")

    # make output directory if doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # specify output
    embedding_tsv = output_dir / f"{args.name}_embeddings.tsv"
    print(f"Output file: {embedding_tsv}")

    # derive the name of input dataset from 'data_ad'
    input_dir = Path(args.data_ad).parent
    print(f"Input dir: {input_dir}")
    params_json = input_dir / "parameters.json"
    with open(params_json, "r") as jsonfile: 
      params = json.load(jsonfile)
    dataset_name = params["dataset_name"]
    print(f"dataset_name: {input_dir}")

    #cmd = ["python", Path.cwd + "/bench/bench.py", "-d "]
    cmd = ["python", Path.cwd + "/bench/bench.py", "-d"]
    print(cmd)

    #print("Running the fetch command: {cmd}")
    #ad = eval(cmd)
    #n_cells, n_features = ad.shape
    #print(f"Got an AnnData with {n_cells} cells and {n_features} features.")
    #print(f"Writing {output_h5ad}.")
    #ad.write_h5ad(output_h5ad)

    #print("Checking output.")
    #stat = Path(output_h5ad).stat()  # raises if file missing
    #print("Size:", stat.st_size, "bytes")
    #print("Created:", dt.datetime.fromtimestamp(stat.st_ctime))

if __name__ == "__main__":
    main()


