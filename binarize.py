#!/usr/bin/env python3

import argparse
import datetime as dt
import sys
from pathlib import Path
import json
import subprocess
import logging
from src.bench.bench import mk_bin_path

def parse_args():
    parser = argparse.ArgumentParser(description='OmniBenchmark module')

    # Required by OmniBenchmark
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--name', type=str, required=True,
                       help='Module name/identifier')
    parser.add_argument('--data_ad', type=str, help='Input dataset')
    parser.add_argument('--embedding_tsv', type=str, help='Embedding TSV')
    parser.add_argument('--phat_tsv', type=str, help='P-hat TSV')
    parser.add_argument('--binarize_method', type=str, help='Binarization Method')
    return parser.parse_args()

def main():
    print("Parsing args.")
    args = parse_args()

    # logging
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "data_ad", "embedding_tsv", "phat_tsv", "binarize_method"):
        print(f"  {k}: {getattr(args, k)}")

    # make output directory if doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # specify output
    bin_tsv = output_dir / f"{args.name}_binarization.tsv"
    print(f"Output file: {bin_tsv}")

    # derive the name of input dataset from 'data_ad'
    input_dir = Path(args.data_ad).parent
    print(f"[data_ad] Input dir: {input_dir}")
    params_json = input_dir / "parameters.json"
    with open(params_json, "r") as jsonfile: 
      params = json.load(jsonfile)
    dataset_name = params["dataset_name"]
    print(f"dataset_name: {dataset_name}")

    # derive the embedding method/dim 'embedding_tsv'
    input_dir = Path(args.embedding_tsv).parent
    print(f"[embedding tsv] Input dir: {input_dir}")
    params_json = input_dir / "parameters.json"
    with open(params_json, "r") as jsonfile: 
      embed_params = json.load(jsonfile)
    print(f"embedding_params: {embed_params["embed_method"]} {embed_params["dim"]}")

    # derive the regression method/dim 'phat_tsv'
    input_dir = Path(args.phat_tsv).parent
    print(f"[phat tsv] Input dir: {input_dir}")
    params_json = input_dir / "parameters.json"
    with open(params_json, "r") as jsonfile: 
      regress_params = json.load(jsonfile)
    print(f"regress_params: {regress_params["regress_method"]}")

    # specify location of Elia's b/m code
    bench_py = Path.cwd() / "src" / "bench" / "bench.py"
    stat = Path(bench_py).stat()

    # manually construct the command
    #python found-engine/src/bench/bench.py bin -c /Users/mark/projects/omb/found/cache \
    #       -z 42 -d jakel -e log_pca 30 -r logit_lbfgs_nol1 -b kmeans

    cache_dir = Path.cwd() / ".." / ".." / ".." / ".found-cache"
    cmd = ["python", str(bench_py), "bin", "-d", dataset_name, 
           "-e", embed_params["embed_method"], str(embed_params["dim"]),
           "-z", str(42), "-c", str(cache_dir), "-r", regress_params["regress_method"],
           "-b", args.binarize_method]
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
    pth = cache_dir / mk_bin_path(dataset_name, None, embed_params["embed_method"], 
                                  embed_params["dim"], regress_params["regress_method"],
                                  args.binarize_method)
    print(bin_tsv)
    print(" --> ")
    print(pth)
    Path(bin_tsv).symlink_to(pth)

    print("Checking output.")
    stat = pth.stat()  # raises if file missing
    print("Size:", stat.st_size, "bytes")
    print("Created:", dt.datetime.fromtimestamp(stat.st_ctime))


if __name__ == "__main__":
    main()


