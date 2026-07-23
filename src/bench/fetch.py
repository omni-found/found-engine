# %%
from collections.abc import Callable
from functools import wraps
from gzip import decompress
from io import BytesIO
from pathlib import Path
from tarfile import TarFile
from urllib.request import urlopen
from zipfile import ZipFile

import anndata as ad
import boto3
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from h5py import File as H5File
from scipy import sparse as sp
from scipy.io import mmread
from stream_unzip import stream_unzip


def wf(pth: Path | None = None) -> Callable[[Callable[[], ad.AnnData]], Callable[[], ad.AnnData]]:
    def pp(adata: ad.AnnData) -> ad.AnnData:
        adata.X = sp.csr_array(adata.X).astype(np.uint)
        adata = adata[adata.X.sum(axis=1) > 0, :]  # ty:ignore[unresolved-attribute]

        assert isinstance(adata.X, sp.csr_array)
        adata = adata[:, adata.X.sum(axis=0) > 0]

        return adata.copy()

    def _(fn: Callable[[], ad.AnnData]) -> Callable[[], ad.AnnData]:
        if pth is not None:

            @wraps(fn)
            def w() -> ad.AnnData:
                adata = pp(ad.read_h5ad(pth) if pth.exists() else fn())
                pth.parent.mkdir(parents=True, exist_ok=True)
                adata.write_h5ad(pth)  # ty:ignore[invalid-argument-type]
                return adata
        else:

            @wraps(fn)
            def w() -> ad.AnnData:
                adata = pp(fn())
                return adata

        return w

    return _


@wf()
def get_jakel() -> ad.AnnData:
    mtx = pd.read_csv(
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE118257&format=file&file=GSE118257_MSCtr_snRNA_ExpressionMatrix_R.txt.gz",
        sep="\t",
    ).T
    return ad.AnnData(
        mtx,
        obs=pd.read_csv(
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE118257&format=file&file=GSE118257_MSCtr_snRNA_FinalAnnotationTable.txt.gz",
            sep="\t",
        )
        .set_index("Detected")
        .loc[mtx.index],
    )


@wf()
def get_schirmer() -> ad.AnnData:
    with urlopen("https://cells.ucsc.edu/ms/rawMatrix.zip") as r:
        h = BytesIO(r.read())
    f = ZipFile(h)
    adata = ad.AnnData(
        sp.csc_array(mmread(BytesIO(f.read("matrix.mtx")))).T,
        var=pd.read_csv(BytesIO(f.read("genes.tsv")), sep="\t", header=None)
        .rename(columns={0: "ENSEMBL", 1: "SYMBOL"})
        .set_index("ENSEMBL"),
        obs=pd.DataFrame(index=f.read("barcodes.tsv").decode().strip().split("\n")),
    )
    adata.obs = pd.read_csv(BytesIO(f.read("meta.txt")), sep="\t").set_index("cell").loc[adata.obs.index]

    return adata


@wf()
def get_absinta() -> ad.AnnData:
    mtx = pd.read_csv(
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE180759&format=file&file=GSE180759_expression_matrix.csv.gz"
    ).T
    adata = ad.AnnData(
        mtx,
        obs=pd.read_csv(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE180nnn/GSE180759/suppl/GSE180759_annotation.txt.gz", sep="\t"
        )
        .set_index("nucleus_barcode")
        .loc[mtx.index],
    )
    return adata


@wf()
def get_macnair() -> ad.AnnData:
    with urlopen("https://zenodo.org/records/8338963/files/ms_lesions_snRNAseq_cleaned_counts_matrix_2023-09-12.mtx.gz") as r:
        mtx = sp.csc_array(mmread(BytesIO(decompress(r.read())))).T
    return ad.AnnData(
        mtx,
        obs=pd.read_csv("https://zenodo.org/records/8338963/files/ms_lesions_snRNAseq_col_data_2023-09-12.txt.gz").set_index(
            "cell_id"
        ),
        var=pd.read_csv("https://zenodo.org/records/8338963/files/ms_lesions_snRNAseq_row_data_2023-09-12.txt.gz").set_index(
            "gene_id"
        ),
    )


@wf()
def get_fournier() -> ad.AnnData:
    adatas = []
    id_map = dict()
    for gsm, smp in [
        ("GSM5973615", "EAE1"),
        ("GSM5973616", "EAE2"),
        ("GSM5973617", "CTL1"),
        ("GSM5973620", "EAE3"),
        ("GSM5973621", "CTL2"),
        ("GSM5973622", "CTL3"),
    ]:
        with urlopen(f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gsm}&format=file&file={gsm}_{smp}_total.tar.gz") as r:
            f = TarFile(fileobj=BytesIO(decompress(r.read())))
            mtx_f = f.extractfile(f"{smp}_total/filtered_feature_bc_matrix/matrix.mtx.gz")
            obs_f = f.extractfile(f"{smp}_total/filtered_feature_bc_matrix/barcodes.tsv.gz")
            var_f = f.extractfile(f"{smp}_total/filtered_feature_bc_matrix/features.tsv.gz")
            assert mtx_f is not None and obs_f is not None and var_f is not None
            ids = decompress(obs_f.read()).decode().strip().split("\n")
            adatas.append(
                ad.AnnData(
                    sp.csc_array(mmread(BytesIO(decompress(mtx_f.read())))).T,
                    obs=pd.DataFrame(index=[f"{smp}_{barcode}" for barcode in ids]),
                    var=pd.read_csv(var_f, sep="\t", header=None, compression="gzip")
                    .rename(columns={0: "ENSEMBL", 1: "SYMBOL"})
                    .set_index("ENSEMBL"),
                )
            )
            id_map[smp] = set(ids)
    adata = ad.concat(adatas)
    obs = pd.read_csv(
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE199460&format=file&file=GSE199460_cell_annotation.meta_data.total_cells.csv.gz"
    )
    new_idx = []
    fixed_col = []
    for smp, barcode in obs.index.str.split("_"):
        if barcode in id_map[smp]:
            new_idx.append(f"{smp}_{barcode}")
            fixed_col.append(False)
        else:
            possible_fixes = [k for k in id_map if barcode in id_map[k]]

            if smp == "CTL2":
                fix = "CTL3"
            elif smp == "CTL3":
                fix = "CTL2"
            else:
                raise AssertionError(f"found improper barcode {smp}_{barcode}, should not occur")
            assert fix in possible_fixes, f"barcode: {smp}_{barcode}\nfixes: {possible_fixes}"

            new_idx.append(f"{fix}_{barcode}")
            fixed_col.append(True)

    obs.index = pd.Index(new_idx)
    obs["id_fixed"] = fixed_col

    adata = adata[obs.index].copy()
    adata.obs = obs.loc[adata.obs.index]

    return adata


def dl_SEAAD_pfc(pth: str):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file("sea-ad-single-cell-profiling", "PFC/RNAseq/SEAAD_A9_RNAseq_final-nuclei.2024-02-13.h5ad", pth)


def dl_SEAAD_mtg(pth: str):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file("sea-ad-single-cell-profiling", "MTG/RNAseq/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad", pth)


def dl_SEAAD_imm(pth: str):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file(
        "sea-ad-single-cell-profiling",
        "Microglia-and-Immune-for-AAIC/SEA-AD_Microglia-and-Immune_multi-regional_final-nuclei_AAIC-pre-release.2025-07-24.h5ad",
        pth,
    )


@wf()
def get_vanhove() -> ad.AnnData:
    base_url = "https://www.brainimmuneatlas.org/data_files/toDownload/"
    zpfx = "filtered_gene_bc_matrices_mex/mm10"
    with urlopen(f"{base_url}/filtered_gene_bc_matrices_mex_irf8_fullAggr.zip") as r:
        f = ZipFile(BytesIO(r.read()))
        adata = ad.AnnData(
            sp.csc_array(mmread(BytesIO(f.read(f"{zpfx}/matrix.mtx")))).T,
            var=pd.read_csv(BytesIO(f.read(f"{zpfx}/genes.tsv")), sep="\t", header=None)
            .rename(columns={0: "ENSEMBL", 1: "SYMBOL"})
            .set_index("ENSEMBL"),
            obs=pd.DataFrame(index=f.read(f"{zpfx}/barcodes.tsv").decode().strip().split("\n")),
        )

    obs = pd.read_csv(f"{base_url}/annot_JP34353637.csv").set_index("cell")
    adata = adata[obs.index].copy()
    adata.obs = obs.loc[adata.obs.index]

    return adata


@wf()
def get_vlaminck() -> ad.AnnData:
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/?format=file&"
    adatas = dict()
    for meta, smp, data in [
        (
            f"{base_url}acc=GSE212078&file=GSE212078%5Fmetadata%5FNaive%5Fmicroglia%5Finfected%5FMicroglia%5FInfected%5FCD45hi%5Faggregate%5FK3%2D4%2D5%2Ecsv.gz",
            "brain",
            [
                (
                    f"{base_url}acc=GSM6508951&file=GSM6508951%5FNaive%5FMicroglia%5Fwhole%5Fbrain%5FK3%5Fraw%5Ffeature%5Fbc%5Fmatrix%2Eh5",
                    "1",
                ),
                (
                    f"{base_url}acc=GSM6508952&file=GSM6508952%5FInfected%5FMicroglia%5Fwhole%5Fbrain%5FK4%5Fraw%5Ffeature%5Fbc%5Fmatrix%2Eh5",
                    "2",
                ),
                (
                    f"{base_url}acc=GSM6508953&file=GSM6508953%5FInfected%5FCD45hi%5Fwhole%5Fbrain%5FK5%5Ffiltered%5Ffeature%5Fbc%5Fmatrix%2Eh5",
                    "3",
                ),
            ],
        ),
        (
            f"{base_url}acc=GSE212078&file=GSE212078%5Fmetadata%5FInfected%5FCP%5FYFP%5FK14%5FNaive%5FCP%5Faggregate%2Ecsv.gz",
            "choroid",
            [
                (
                    f"{base_url}acc=GSM6508955&file=GSM6508955%5FNaive%5FCP%5FK11%5Fraw%5Ffeature%5Fbc%5Fmatrix%2Eh5",
                    "2",
                ),
                (
                    f"{base_url}acc=GSM6508954&file=GSM6508954%5FInfected%5FCP%5FYFP%5FK14%5Fraw%5Ffeature%5Fbc%5Fmatrix%2Eh5",
                    "1",
                ),
            ],
        ),
    ]:
        adict = dict()
        for url, key in data:
            with urlopen(url) as r:
                m = H5File(BytesIO(r.read()), "r")["matrix"]
                adict[key] = ad.AnnData(
                    sp.csc_array((m["data"], m["indices"], m["indptr"]), shape=m["shape"]).T,
                    var=pd.DataFrame({"SYMBOL": m["features"]["name"], "ENSEMBL": m["features"]["id"]}).set_index("ENSEMBL"),
                    obs=pd.DataFrame({"cell": m["barcodes"]}).set_index("cell"),
                )
        adata = ad.concat(adict, index_unique="_")

        obs = (
            pd.read_csv(meta)
            .set_index("cell")
            .merge(
                adata.obs,  # ty:ignore[invalid-argument-type]
                left_index=True,
                right_index=True,
            )
        )
        adata = adata[obs.index].copy()
        adata.obs = obs.loc[adata.obs.index]

        adatas[smp] = adata

    return ad.concat(adatas, label="tissue", index_unique="_")


@wf()
def get_andries() -> ad.AnnData:
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/?format=file&acc="
    adatas = dict()
    for url, key, sfx in [
        (f"{base_url}GSM7336928&file=GSM7336928%5FNaive%5Fretina%5FLM1%5Fraw%5Ffeature%5Fbc%5Fmatrix%2Eh5", "naive", "_1"),
        (f"{base_url}GSM7336929&file=GSM7336929%5FInjured%5Fretina%5FLM2%5Fraw%5Ffeature%5Fbc%5Fmatrix%2Eh5", "injured", "_2"),
    ]:
        with urlopen(url) as r:
            m = H5File(BytesIO(r.read()), "r")["matrix"]
            adatas[key] = ad.AnnData(
                sp.csc_array((m["data"], m["indices"], m["indptr"]), shape=m["shape"]).T,
                var=pd.DataFrame({"SYMBOL": m["features"]["name"], "ENSEMBL": m["features"]["id"]}).set_index("ENSEMBL"),
                obs=pd.DataFrame({"cell": pd.Series(m["barcodes"]).str.decode("utf8") + sfx}).set_index("cell"),
            )
    adata = ad.concat(adatas, label="sample")

    obs = (
        pd.read_csv("https://www.brainimmuneatlas.org/data_files/toDownload/annot_retina.csv")
        .set_index("cell")
        .merge(
            adata.obs,  # ty:ignore[invalid-argument-type]
            left_index=True,
            right_index=True,
        )
    )
    adata = adata[obs.index].copy()
    adata.obs = obs.loc[adata.obs.index]

    return adata


@wf()
def get_wilk() -> ad.AnnData:
    with urlopen("https://datasets.cellxgene.cziscience.com/89c999bd-2ba9-4281-9d22-4261347c5c78.h5ad") as r:
        return ad.read_h5ad(
            BytesIO(r.read()),  # ty:ignore[invalid-argument-type]
        ).raw.to_adata()


def dl_oesinghaus(pth: Path):
    with urlopen("https://parse-wget.s3.us-west-2.amazonaws.com/10m/10M_PBMC_12donor_90cytokines_h5ad.zip") as r:
        for fn, _, chunks in stream_unzip(r):
            if fn != b"Parse_10M_PBMC_cytokines.h5ad":
                # consume the iterator without saving it
                for chunk in chunks:
                    pass
            with pth.open("wb") as f:
                for chunk in chunks:
                    f.write(chunk)


# %%
if __name__ == "__main__":
    pass
