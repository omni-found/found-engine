from collections.abc import Iterator
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from pstats import Stats
from re import Pattern, compile
from sys import argv

from memray import FileReader


def proc_match(pat: Pattern, p: Path) -> None | tuple:
    pat_m = pat.search(str(p))
    if pat_m is None:
        return None

    mr_path = Path(pat_m.string, "memray.bin")
    if not mr_path.exists():
        return None

    peak_mem = FileReader(mr_path).metadata.peak_memory
    total_time = Stats(f"{pat_m.string}/cprofile.pstats").get_stats_profile().total_tt

    base = (pat_m["dataset"], pat_m["grp"] or "", str(peak_mem), str(total_time), pat_m["emb"], pat_m["k"])

    if pat_m["bin"] is not None:
        return (*base, pat_m["reg"], pat_m["bin"])
    if pat_m["reg"] is not None:
        return (*base, pat_m["reg"])

    return base


if __name__ == "__main__":
    base_dir = Path(argv[1])
    pat = compile(
        rf"^{str(base_dir)}/(?P<dataset>[^/]+)/(?P<emb>[^/]+)/k<?\=(?P<k>[0-9]+)((/(?P<reg>(logit|svm|rf)_[^/]+))?(/(?P<bin>gmm|kmeans))?(/(?P<grp>[^/]+))?)$"
    )
    header = ["dataset", "grp", "peak_mem", "total_time", "emb", "k", "reg", "bin"]
    with Pool() as p:
        stats: Iterator[tuple] = filter(
            lambda o: o is not None, p.imap_unordered(partial(proc_match, pat), base_dir.glob("**"))
        )

        fst = next(stats)
        print("\t".join(header[: len(fst)]), flush=True)
        print("\t".join(fst), flush=True)
        print("\n".join("\t".join(o) for o in stats))
