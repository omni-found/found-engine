#!/usr/bin/env python3

import argparse
import datetime as dt
import sys
from pathlib import Path
import json
import subprocess
import logging
from src.bench.bench import mk_emb_path

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
    embedding_tsv = output_dir / f"{args.name}_embedding.tsv"
    print(f"Output file: {embedding_tsv}")

    # derive the name of input dataset from 'data_ad'
    input_dir = Path(args.data_ad).parent
    print(f"Input dir: {input_dir}")
    params_json = input_dir / "parameters.json"
    with open(params_json, "r") as jsonfile: 
      params = json.load(jsonfile)
    dataset_name = params["dataset_name"]
    print(f"dataset_name: {dataset_name}")

    # specify location of Elia's b/m code
    bench_py = Path.cwd() / "src" / "bench" / "bench.py"
    stat = Path(bench_py).stat()

    # manually construct the command
    #python bench.py emb -c /data/mark/found-cache -z 42 -d jakel -e log_pca
    cache_dir = Path.cwd() / ".." / ".." / ".." / ".found-cache"
    cmd = ["python", str(bench_py), "emb",
           "-d", dataset_name, "-e", args.embed_method, str(args.dim),
           "-z", str(42), "-c", str(cache_dir)]
    print("Command to run:")
    print(" ".join(cmd))

    # run the process in a way that captures the deets
    try:
      result = subprocess.run(cmd, check=True, capture_output=True, text=True)
      logging.info("stdout: %s", result.stdout)
      logging.info("stderr: %s", result.stderr)
    except subprocess.CalledProcessError as e:
      logging.error("Command failed with return code %s", e.returncode)
      logging.error("stdout: %s", e.stdout)
      logging.error("stderr: %s", e.stderr)
      logging.error("cmd: %s", e.cmd)

    # derive location of output (to create symlink to)
    print("Creating symlink to cache.")
    pth = cache_dir / mk_emb_path(dataset_name, None, args.embed_method, args.dim) 
    print(embedding_tsv)
    print(" --> ")
    print(pth)
    Path(embedding_tsv).symlink_to(pth)

    print("Checking output.")
    stat = pth.stat()  # raises if file missing
    print("Size:", stat.st_size, "bytes")
    print("Created:", dt.datetime.fromtimestamp(stat.st_ctime))


if __name__ == "__main__":
    main()


