from dataclasses import dataclass
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import Annotated

from cappa import Arg, ArgAction, Subcommands, command, parse

from found.seed import get_seed, set_seed

LOGGER = getLogger(__name__)
# set up logger handler if running as app
if __name__ == "__main__":
    hd = StreamHandler()
    hd.setFormatter(Formatter("%(levelname)s - %(asctime)s - %(message)s"))
    LOGGER.addHandler(hd)


@dataclass(frozen=True)
class RunBaseArgs:
    # value name / long / default need to be specified manually due to seeming cappa bug w/ inheritance?
    cache: Annotated[Path, Arg(short="c", value_name="CACHE_PATH")]
    skip_if_exists: Annotated[bool, Arg(short="s", long="skip", default=False)]
    catch_on_err: Annotated[bool, Arg(short="f", long="force", default=False)]
    seed: Annotated[int | None, Arg(short="z", value_name="SEED", default=None)]


@dataclass(frozen=True)
class RunEmbArgs(RunBaseArgs):
    datasets: Annotated[list[str], Arg(short="d", action=ArgAction.append, value_name="DATASET\\[/GROUP]")]

    emb_methods: Annotated[list[tuple[str, int]], Arg(short="e", action=ArgAction.append, value_name="EMB_METHOD EMB_K")]


@dataclass(frozen=True)
class RunRegArgs(RunEmbArgs):
    reg_methods: Annotated[list[str], Arg(short="r", action=ArgAction.append, value_name="REG_METHOD")]


@dataclass(frozen=True)
class RunBinArgs(RunRegArgs):
    bin_methods: Annotated[list[str], Arg(short="b", action=ArgAction.append, value_name="BIN_METHOD")]


# fmt: off
@command(name="emb")
@dataclass(frozen=True)
class RunEmbArgs(RunEmbArgs):
    pass
@command(name="reg")
@dataclass(frozen=True)
class RunRegArgs(RunRegArgs):
    pass
@command(name="bin")
@dataclass(frozen=True)
class RunBinArgs(RunBinArgs):
    pass
# fmt: on


@command(name="fcf")
@dataclass(frozen=True)
class FromCmdFileArgs(RunBaseArgs):
    cmd_file: Annotated[Path, Arg(short="p")]


@command(name="ldts")
@dataclass(frozen=True)
class ListDatasets:
    pass


@command(name="qdts")
@dataclass(frozen=True)
class QueryDataset:
    dataset: Annotated[str, Arg(short="d")]
    cache: Annotated[Path, Arg(short="c")]

    min_n: Annotated[int, Arg(short="m")] = 0


@command(name="lemb")
@dataclass(frozen=True)
class ListEmbs:
    cachable: Annotated[bool, Arg(short="o")] = False


@command(name="lreg")
@dataclass(frozen=True)
class ListRegs:
    pass


@command(name="lbin")
@dataclass(frozen=True)
class ListBins:
    pass


@command(name=Path(__file__).stem)
@dataclass(frozen=True)
class Cmd:
    cmd: Subcommands[
        RunEmbArgs | RunRegArgs | RunBinArgs | ListDatasets | QueryDataset | ListEmbs | ListRegs | ListBins | FromCmdFileArgs
    ]
    log_level: Annotated[
        int | None,
        Arg(
            short="l",
            parse=lambda e: (
                next(
                    (
                        lvl_i
                        for lvl_s, lvl_i in [
                            ("debug", DEBUG),
                            ("info", INFO),
                            ("warning", WARNING),
                            ("error", ERROR),
                            ("critical", CRITICAL),
                        ]
                        if lvl_s[: len(e)] == e.lower()
                    ),
                    None,
                )
                or int(e)
            ),
        ),
    ] = None


# parse args before remaining imports for quick help message load
CMD = None
if __name__ == "__main__":
    CMD = parse(Cmd, completion=False)
    if CMD.log_level is not None:
        LOGGER.setLevel(CMD.log_level)

# ruff: noqa: E402
from collections.abc import Callable, Mapping
from cProfile import Profile
from functools import partial
from itertools import chain
from logging import Formatter, StreamHandler, getLogger
from traceback import format_exception
from typing import Never, Protocol, Self

import anndata as ad
import numpy as np
import pandas as pd
from memray import FileDestination, Tracker
from scipy import sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from found import methods as m
from found.types import BoolArr, FloatMtx, MatrixLike, NumArr


def analytic_pearson_res(X: MatrixLike, theta: float = 100, clip: float | None = None) -> FloatMtx:
    sums_genes = X.sum(axis=0)
    sums_cells = X.sum(axis=1)
    sum_total = np.sum(sums_genes)

    mu = np.outer(sums_cells, sums_genes) / sum_total

    diff = X - mu
    res = diff / np.sqrt(mu + mu**2 / theta)

    if clip is None:
        assert X.shape is not None
        clip = np.sqrt(X.shape[0])
        assert clip is not None

    res = np.clip(res, -clip, clip)

    return res


def glm_pca(X: MatrixLike, k: int) -> FloatMtx:
    from glmpca.glmpca import glmpca

    if not isinstance(X, np.ndarray):
        X = X.todense()
    return glmpca(X.T, k)["factors"]


def wrap_harmony(Z: Callable[[MatrixLike, int], FloatMtx], batch: pd.Series, X: MatrixLike, k: int) -> FloatMtx:
    from rpy2.robjects import DataFrame as rDataFrame
    from rpy2.robjects import FloatVector, StrVector, r
    from rpy2.robjects.packages import importr

    Z_mtx = Z(X, k)

    hm = importr("harmony")
    idx_rpy = StrVector(batch.index.to_numpy())
    Z_rpy = r.matrix(
        FloatVector(np.ravel(Z_mtx, "F")),
        nrow=Z_mtx.shape[0],
        ncol=Z_mtx.shape[1],
    )  # ty:ignore[call-non-callable]
    Z_rpy.rownames = idx_rpy
    df_rpy = rDataFrame({"b": StrVector(batch.to_numpy())})
    df_rpy.rownames = idx_rpy

    return np.array(hm.RunHarmony(Z_rpy, df_rpy, "b", verbose=False))


def scvi(batch: pd.Series, X: MatrixLike, k: int) -> FloatMtx:
    from scvi.model import SCVI

    adata = ad.AnnData(sp.csr_matrix(X), obs=pd.DataFrame({"b": batch}))

    SCVI.setup_anndata(adata, batch_key="b")
    vae = SCVI(adata, n_latent=k)
    vae.train()
    return vae.get_latent_representation()  # ty:ignore[invalid-return-type]


def wrap_lemur(N: Callable[[MatrixLike], MatrixLike], align: bool | str, batch: pd.Series, X: MatrixLike, k: int) -> FloatMtx:
    from pylemur.tl import LEMUR

    adata = ad.AnnData(N(X), obs=pd.DataFrame({"b": batch}))

    model = LEMUR(adata, design="~ b", n_embedding=k)
    model.fit()
    if align:
        model.align_with_harmony()

    assert model.embedding is not None

    return model.embedding


def pca(X: MatrixLike, k: int, f: Callable[[MatrixLike], MatrixLike]) -> FloatMtx:
    return m.run_pca(X, k, f)


def nmf(X: MatrixLike, k: int, f: Callable[[MatrixLike], MatrixLike]) -> FloatMtx:
    return m.run_nmf(X, k, f)[0]


def shuf_reg[M](reg_fn: Callable[[FloatMtx, BoolArr], tuple[NumArr, M]]) -> Callable[[FloatMtx, BoolArr], tuple[NumArr, M]]:
    def w(Z: FloatMtx, V: BoolArr) -> tuple[NumArr, M]:
        seed = get_seed()
        V_shuf = np.random.default_rng(seed).permuted(V)
        LOGGER.debug(f"shuf_reg : shuffling array {V} (w/ seed - {seed}), yielding: {V_shuf}")
        return reg_fn(Z, V_shuf)

    return w


DATA_CONF: dict[
    str,
    tuple[
        str,
        Callable[[pd.DataFrame], BoolArr],
        Callable[[pd.DataFrame], BoolArr],
        Callable[[pd.DataFrame], pd.Series],
        Callable[[pd.DataFrame], pd.Series],
    ],
] = {
    "absinta": (
        "absinta",
        lambda meta: meta["pathology"].isin(["MS_lesion_core", "control_white_matter"]).to_numpy(),
        lambda meta: (meta["pathology"] != "control_white_matter").to_numpy(),
        lambda meta: meta["cell_type"],
        lambda meta: meta.index.to_series().str.split("-", expand=True)[1],
    ),
    "andries": (
        "andries",
        lambda meta: np.full((meta.shape[0],), True),
        lambda meta: (meta["condition"] != "Naive retina").to_numpy(),
        lambda meta: meta["cell_type"],
        lambda meta: meta.index.to_series().str.split("-", expand=True)[1],
    ),
    "fournier": (
        "fournier",
        lambda meta: np.full((meta.shape[0],), True),
        lambda meta: (meta["condition"] != "C").to_numpy(),
        lambda meta: meta["cell_type"],
        lambda meta: meta["batch"],
    ),
    "jakel": (
        "jakel",
        lambda meta: meta["Lesion"].isin(["Ctrl", "A"]).to_numpy(),
        lambda meta: (meta["Condition"] != "Ctrl").to_numpy(),
        lambda meta: meta["Celltypes"],
        lambda meta: meta["Sample"],
    ),
    "macnair": (
        "macnair",
        lambda meta: meta["lesion_type"].isin(["GM", "WM", "AL", "GML"]).to_numpy(),
        lambda meta: (meta["diagnosis"] != "CTR").to_numpy(),
        lambda meta: meta["type_fine"],
        lambda meta: meta["sample_id_anon"],
    ),
    "schirmer": (
        "schirmer",
        lambda meta: meta["stage"].isin(["Control", "Acute/Chronic active"]).to_numpy(),
        lambda meta: (meta["diagnosis"] != "Control").to_numpy(),
        lambda meta: meta["cell_type"],
        lambda meta: meta["sample"],
    ),
    "vanhove": (
        "vanhove",
        lambda meta: np.full((meta.shape[0],), True),
        lambda meta: ~meta["sample"].str.startswith("WT").to_numpy(),
        lambda meta: meta["sample"].str.extract("(CP-BAM|microglia)", expand=False),
        lambda meta: meta.index.to_series().str.split("-", expand=True)[1],
    ),
    "wilk": (
        "wilk",
        lambda meta: np.full((meta.shape[0],), True),
        lambda meta: (meta["disease"] != "normal").to_numpy(),
        lambda meta: meta["cell.type.fine"],
        lambda meta: meta.index.to_series().str.split(".", expand=True)[0],
    ),
    "oesinghaus_IFNg": (
        "oesinghaus",
        lambda meta: meta["cytokine"].isin(["PBS", "IFN-gamma"]).to_numpy(),
        lambda meta: (meta["treatment"] != "PBS").to_numpy(),
        lambda meta: meta["cell_type"],
        lambda meta: meta.index.to_series().str.split("__", expand=True)[1],
    ),
    "oesinghaus_TNFa": (
        "oesinghaus",
        lambda meta: meta["cytokine"].isin(["PBS", "TNF-alpha"]).to_numpy(),
        lambda meta: (meta["treatment"] != "PBS").to_numpy(),
        lambda meta: meta["cell_type"],
        lambda meta: meta.index.to_series().str.split("__", expand=True)[1],
    ),
}

# add logic applied to all elements (e.g. string cleaning for some columns)
for k in DATA_CONF:
    DATA_CONF[k] = (
        DATA_CONF[k][0],
        DATA_CONF[k][1],
        DATA_CONF[k][2],
        # gross but necessary to avoid creating temp variables which are mutated
        (lambda o: lambda meta: o(meta).str.replace("/", "+").str.replace(" ", "_"))(DATA_CONF[k][3]),
        DATA_CONF[k][4],
    )


# create protocols since Callable does not allow specifying keyword only arguments
# fmt: off
class NaiveEmbFn(Protocol):
    def __call__(self, X: MatrixLike, k: int) -> FloatMtx: ...
class AwareEmbFn(Protocol):
    def __call__(self, batch: pd.Series, X: MatrixLike, k: int) -> FloatMtx: ...
# fmt: on


EMB_NAIVE_CONF: dict[str, tuple[NaiveEmbFn, bool]] = {
    "log_pca": (partial(pca, f=m.vst_shiftlog), True),
    "res_pca": (partial(pca, f=analytic_pearson_res), True),
    "sd_log_pca": (partial(pca, f=lambda X: m.scale_sd(m.vst_shiftlog(X))), True),
    "sd_res_pca": (partial(pca, f=lambda X: m.scale_sd(analytic_pearson_res(X))), True),
    "sd_nmf": (partial(nmf, f=m.scale_sd), False),
    "rs_nmf": (partial(nmf, f=m.scale_rs), False),
    "log_nmf": (partial(nmf, f=m.vst_shiftlog), False),
    "sd_rs_nmf": (partial(nmf, f=lambda X: m.scale_sd(m.scale_rs(X))), False),
    "sd_log_nmf": (partial(nmf, f=lambda X: m.scale_sd(m.vst_shiftlog(X))), False),
    "glm_pca": (glm_pca, False),
}

EMB_AWARE_CONF: dict[str, AwareEmbFn] = (
    {f"{k}_harmony": partial(wrap_harmony, f) for k, (f, _) in EMB_NAIVE_CONF.items()}
    | {
        f"{k}_{'lemur' if align else 'mcPCA'}": partial(wrap_lemur, f, align)
        for k, f in {
            "log": m.vst_shiftlog,
            "res": analytic_pearson_res,
            "sd_log": lambda X: m.scale_sd(m.vst_shiftlog(X)),
            "sd_res": lambda X: m.scale_sd(analytic_pearson_res(X)),
        }.items()
        for align in [True, False]
    }
    | {"scVI": scvi}
)

type RegModel = LogisticRegression | SVC | LinearSVC | RandomForestClassifier
type RegFn = Callable[[FloatMtx, BoolArr], tuple[NumArr, RegModel]]
REG_CONF: Mapping[str, RegFn] = (
    {
        f"logit_{arg['solver']}_{'nol1' if arg['C'] == np.inf else f'l1c{arg["C"]}'}": lambda Z, V: m.reg_logit(Z, V, arg)
        for arg in [
            *({"solver": s, "max_iter": int(1e4), "C": np.inf} for s in ["lbfgs", "newton-cg", "newton-cholesky"]),
            *({"solver": "saga", "max_iter": int(1e5), "C": c, "l1_ratio": 1} for c in [0.01, 1.0, 100, np.inf]),
        ]
    }
    | {f"svm_libsvm_{krn}": lambda Z, V: m.reg_svm(Z, V, {"kernel": krn}) for krn in ["linear", "poly", "rbf", "sigmoid"]}
    | {
        f"svm_liblinear_l1c{c}": lambda Z, V: m.reg_lsvm(Z, V, {"C": c, "penalty": "l1", "max_iter": int(1e7)})
        for c in [0.01, 1.0, 100]
    }
    | {f"rf_{c}": lambda Z, V: m.reg_rf(Z, V, {"criterion": c}) for c in ["gini", "entropy", "log_loss"]}
)
# add shuffled outputs
REG_CONF = REG_CONF | {f"{k}_shuf": shuf_reg(v) for k, v in REG_CONF.items()}

type BinFn = Callable[[NumArr, BoolArr], BoolArr]
BIN_CONF: dict[str, BinFn] = {"kmeans": m.bin_kmeans, "gmm": m.bin_gmm}

type DataTup = tuple[Callable[[], sp.csr_array], BoolArr, pd.Series]


def mint_data_acq(dataset: str, cache_path: Path) -> Callable[[str | None], DataTup]:
    ad_name, proc, condi, group, batch = DATA_CONF[dataset]
    ad_pth = cache_path / "data" / f"{ad_name}.h5ad"

    if not ad_pth.exists():
        msg = f"{dataset} : could not find h5ad file @ {ad_pth}"
        LOGGER.critical(msg)
        raise ValueError(msg)

    LOGGER.debug(f"{dataset} : reading at path {ad_pth} as backed anndata")
    adata_b = ad.read_h5ad(ad_pth, backed="r")
    obs_full = adata_b.obs
    mtx_full = adata_b.X

    assert isinstance(obs_full, pd.DataFrame)
    assert isinstance(mtx_full, ad.abc.CSRDataset)

    LOGGER.debug(f"{dataset} : running initial processing filter")
    proc_filt = proc(obs_full)
    obs_filt = obs_full.loc[proc_filt]
    assert isinstance(obs_filt, pd.DataFrame)

    mtx_cache: sp.csr_array | None = None

    LOGGER.debug(f"{dataset} : accessing condition and batch data")
    V = condi(obs_filt)
    batch_series = batch(obs_filt)

    def get(V: BoolArr, batch_series: pd.Series, grp: str | None) -> tuple[Callable[[], sp.csr_array], BoolArr, pd.Series]:
        grp_info = None

        if grp is not None:
            LOGGER.debug(f"{dataset}/{grp} : filtering condition and batch data to group only")
            grp_mask = group(obs_filt).eq(grp).to_numpy()
            n = np.sum(grp_mask)

            if n == 0:
                msg = f"{dataset}/{grp} : no cells found for group"
                LOGGER.critical(msg)
                raise ValueError(msg)

            V = V[grp_mask]
            batch_series = batch_series.loc[grp_mask]
            assert isinstance(batch_series, pd.Series)

            grp_info = (grp, n, grp_mask)

        def get_X() -> sp.csr_array:
            nonlocal mtx_cache
            if grp_info is None:
                LOGGER.debug(f"{dataset} : reading full data matrix (n = {obs_filt.shape[0]}) to memory from backed anndata")

                mtx_cache = sp.csr_array(mtx_full[proc_filt, :])
                return mtx_cache

            grp, n, grp_mask = grp_info
            if mtx_cache is not None:
                LOGGER.debug(f"{dataset}/{grp} : reading group data matrix (n = {n}) to memory from in-memory full matrix")
                return mtx_cache[grp_mask, :]

            # recompute filter because successive views are not supported
            full_grp_mask = np.copy(proc_filt)
            full_grp_mask[proc_filt] = grp_mask
            # sanity check
            assert np.sum(full_grp_mask) == n

            LOGGER.debug(f"{dataset}/{grp} : reading group data matrix (n = {n}) to memory from backed anndata")
            return sp.csr_array(mtx_full[full_grp_mask, :])

        return get_X, V, batch_series

    return partial(get, V, batch_series)


def is_cacheable(emb_meth: str) -> bool:
    return (emb_meth in EMB_NAIVE_CONF) and EMB_NAIVE_CONF[emb_meth][1]


def mk_emb_path(dataset: str, grp: str | None, emb_meth: str, k: int) -> Path:
    return Path(
        "emb",
        dataset,
        emb_meth,
        f"k{'<=' if is_cacheable(emb_meth) else '='}{k}",
        *([] if grp is None else [grp]),
        "out.tsv",
    )


def mk_reg_path(dataset: str, grp: str | None, emb_meth: str, k: int, reg_meth: str) -> Path:
    return Path("reg", dataset, emb_meth, f"k={k}", reg_meth, *([] if grp is None else [grp]), "out.tsv")


def mk_bin_path(dataset: str, grp: str | None, emb_meth: str, k: int, reg_meth: str, bin_meth: str) -> Path:
    return Path("bin", dataset, emb_meth, f"k={k}", reg_meth, bin_meth, *([] if grp is None else [grp]), "out.tsv")


def get_emb_mtx(cache: Path, dataset: str, grp: str | None, emb_meth: str, k: int, check_idx: pd.Index) -> FloatMtx:
    in_pth = cache / mk_emb_path(dataset, grp, emb_meth, k)

    if not in_pth.exists():
        if not is_cacheable(emb_meth):
            msg = f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : could not find embedding file ({in_pth})"
            LOGGER.critical(msg)
            raise ValueError(msg)

        LOGGER.info(
            f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : could not find exact k match for embedding, checking for possible higher-dimension spaces to subset"
        )
        search_pth = in_pth.parent.parent
        if grp is not None:
            search_pth = search_pth.parent
        for in_pth in search_pth.glob("k<=*/"):
            LOGGER.debug(f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : checking if {in_pth} works")
            if int(in_pth.stem.split("<=", maxsplit=1)[1]) >= k:
                if grp is not None:
                    in_pth /= grp
                in_pth /= "out.tsv"
                if in_pth.exists():
                    break
        else:
            msg = f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : could not find higher-dimension embedding to subset"
            LOGGER.critical(msg)
            raise ValueError(msg)

    LOGGER.debug(f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : reading embedding matrix from {in_pth}")
    df = pd.read_csv(in_pth, sep="\t", usecols=range(k + 1)).set_index("cell")

    if np.any(df.index != check_idx):
        msg = f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : embedding output index does not match dataset index"
        LOGGER.critical(msg)
        raise ValueError(msg)

    return df.to_numpy()


@dataclass(frozen=True, init=False, kw_only=True)
class BaseConf:
    cache: Path
    skip: bool
    catch: bool
    seed: int | None

    def __post_init__(self):
        LOGGER.info(f"{type(self).__name__} - init : calling `found.set_seed` with {self.seed}")
        set_seed(self.seed)
        # sanity check
        assert get_seed() == self.seed

        with pd.option_context("display.max_rows", 1):
            LOGGER.debug(f"{type(self).__name__} - init : {str(self).replace('\n', ' ')}")


@dataclass(frozen=True, kw_only=True)
class RunEmbConf(BaseConf):
    type EmbDict = dict[
        tuple[str, str | None],
        tuple[
            Callable[[], DataTup],
            dict[str, tuple[NaiveEmbFn, list[int]]],
            dict[str, tuple[AwareEmbFn, list[int]]],
        ],
    ]
    emb_runs: EmbDict

    @classmethod
    def mk_dict(cls, cache: Path, emb_runs: dict[str, list[tuple[str, int]]]) -> EmbDict:

        dataset_grps: dict[str, set[str | None]] = dict()
        for dataset in emb_runs.keys():
            base, *grp = dataset.split("/", maxsplit=1)

            if base not in DATA_CONF:
                msg = f"{dataset} : dataset must be in {list(DATA_CONF.keys())}, but got {base}"
                LOGGER.critical(msg)
                raise ValueError(msg)

            grp = None if len(grp) == 0 else grp[0]
            if base in dataset_grps:
                dataset_grps[base].add(grp)
            else:
                dataset_grps[base] = set((grp,))

        runs: cls.EmbDict = dict()

        for dataset, grps in dataset_grps.items():
            dfn = mint_data_acq(dataset, cache)
            for grp in grps:
                pfn = partial(dfn, grp)
                naive_dict: dict[str, tuple[NaiveEmbFn, list[int]]] = dict()
                aware_dict: dict[str, tuple[AwareEmbFn, list[int]]] = dict()
                for emb_meth, k in emb_runs[f"{dataset}/{grp}" if grp is not None else dataset]:
                    if k < 1:
                        msg = f"{emb_meth}(k={k}) : k must be positive rational but got {k}"
                        LOGGER.critical(msg)
                        raise ValueError(msg)
                    if emb_meth in EMB_NAIVE_CONF:
                        if emb_meth in naive_dict:
                            naive_dict[emb_meth][1].append(k)
                        else:
                            naive_dict[emb_meth] = (EMB_NAIVE_CONF[emb_meth][0], [k])
                    elif emb_meth in EMB_AWARE_CONF:
                        if emb_meth in aware_dict:
                            aware_dict[emb_meth][1].append(k)
                        else:
                            aware_dict[emb_meth] = (EMB_AWARE_CONF[emb_meth], [k])
                    else:
                        msg = f"{emb_meth}(k={k}) : embedding method must be in {list(EMB_NAIVE_CONF.keys() | EMB_AWARE_CONF.keys())} but got {emb_meth}"
                        LOGGER.critical(msg)
                        raise ValueError(msg)
                runs[(dataset, grp)] = (pfn, naive_dict, aware_dict)

        return runs

    @classmethod
    def from_args(cls, args: RunEmbArgs) -> Self:
        LOGGER.debug(f"{cls.__name__}.from_args : got args - {args}")

        return cls(
            emb_runs=cls.mk_dict(cache, {ds: args.emb_methods for ds in args.datasets}),
            cache=args.cache,
            skip=args.skip_if_exists,
            catch=args.catch_on_err,
            seed=args.seed,
        )

    def run(self) -> None:
        err_log_fn = LOGGER.error if self.catch else LOGGER.critical

        for dataset, grp in sorted(
            self.emb_runs.keys(),
            key=lambda x: (tuple(map(ord, x[0])), (-1,) if x[1] is None else tuple(map(ord, x[1]))),
        ):
            dfn, naive_fns, aware_fns = self.emb_runs[(dataset, grp)]

            X_f, _, batch = dfn()
            X = X_f()
            for emb_meth, (efn, ks) in chain(
                naive_fns.items(), ((emb, (partial(fn, batch=batch), ks)) for emb, (fn, ks) in aware_fns.items())
            ):
                for k in ks:
                    out_pth = self.cache / mk_emb_path(dataset, grp, emb_meth, k)

                    if out_pth.exists():
                        if self.skip:
                            LOGGER.info(
                                f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : skipping because output file {out_pth} already exists"
                            )
                            continue
                        LOGGER.warning(
                            f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : output file {out_pth} already exists, will be overwritten"
                        )

                    out_pth.parent.mkdir(parents=True, exist_ok=True)

                    LOGGER.info(f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : running embedding")
                    prof = Profile()
                    try:
                        with Tracker(destination=FileDestination(out_pth.with_name("memray.bin"), overwrite=True)):
                            Z = prof.runcall(efn, X, k)
                    except Exception as exc:
                        err_log_fn(
                            f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : caught exception during embedding, see below for traceback\n{'\n'.join(format_exception(exc))}\n"
                        )
                        if not self.catch:
                            raise exc
                    else:
                        prof.dump_stats(out_pth.with_name("cprofile.pstats"))
                        LOGGER.info(
                            f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : saving embedding to {out_pth}"
                        )
                        pd.DataFrame(Z).set_index(batch.index).to_csv(out_pth, index_label="cell", sep="\t")


@dataclass(frozen=True, kw_only=True)
class RunRegConf(BaseConf):
    type RegDict = dict[
        tuple[str, str | None],
        tuple[
            Callable[[], DataTup],
            dict[tuple[str, int], dict[str, RegFn]],
        ],
    ]
    reg_runs: RegDict

    @classmethod
    def mk_dict(cls, cache: Path, reg_runs: dict[str, dict[tuple[str, int], list[str]]]) -> RegDict:
        emb_dict = RunEmbConf.mk_dict(cache, {ds: list(regs.keys()) for ds, regs in reg_runs.items()})

        reg_fns: dict[str, RegFn] = dict()
        for run in reg_runs.values():
            for regs in run.values():
                for reg_meth in regs:
                    if reg_meth not in REG_CONF:
                        msg = f"{reg_meth} : regression method must be in {list(REG_CONF.keys())} but got {reg_meth}"
                        LOGGER.critical(msg)
                        raise ValueError(msg)
                    reg_fns[reg_meth] = REG_CONF[reg_meth]

        return {
            (ds, grp): (
                fn,
                {
                    (emb, k): {nm: reg_fns[nm] for nm in reg_runs[f"{ds}/{grp}" if grp is not None else ds][(emb, k)]}
                    for emb, k in chain(
                        ((nf, k) for nf, (_, ks) in nc.items() for k in ks),
                        ((af, k) for af, (_, ks) in ac.items() for k in ks),
                    )
                },
            )
            for (ds, grp), (fn, nc, ac) in emb_dict.items()
        }

    @classmethod
    def from_args(cls, args: RunRegArgs) -> Self:
        LOGGER.debug(f"{cls.__name__}.from_args : got args - {args}")

        return cls(
            reg_runs=cls.mk_dict(args.cache, {ds: {emb: args.reg_methods for emb in args.emb_methods} for ds in args.datasets}),
            cache=args.cache,
            skip=args.skip_if_exists,
            catch=args.catch_on_err,
            seed=args.seed,
        )

    def run(self) -> None:
        err_log_fn = LOGGER.error if self.catch else LOGGER.critical

        for (dataset, grp), (dfn, embs) in self.reg_runs.items():
            _, V, batch = dfn()
            for (emb_meth, k), regs in embs.items():
                Z = get_emb_mtx(self.cache, dataset, grp, emb_meth, k, batch.index)
                LOGGER.debug(
                    f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : got embedding matrix of shape {Z.shape}"
                )

                for reg_meth, rfn in regs.items():
                    out_pth = self.cache / mk_reg_path(dataset, grp, emb_meth, k, reg_meth)

                    if out_pth.exists():
                        if self.skip:
                            LOGGER.info(
                                f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k})/{reg_meth} : skipping because output file {out_pth} already exists"
                            )
                            continue
                        LOGGER.warning(
                            f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k})/{reg_meth} : output file {out_pth} already exists, will be overwritten"
                        )

                    out_pth.parent.mkdir(parents=True, exist_ok=True)

                    LOGGER.info(
                        f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k})/{reg_meth} : running regression"
                    )
                    prof = Profile()
                    try:
                        with Tracker(destination=FileDestination(out_pth.with_name("memray.bin"), overwrite=True)):
                            Y, mod = prof.runcall(rfn, Z, V)
                    except Exception as exc:
                        err_log_fn(
                            f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k})/{reg_meth} : caught exception during regression, see below for traceback\n{'\n'.join(format_exception(exc))}\n"
                        )
                        if not self.catch:
                            raise exc
                    else:
                        prof.dump_stats(out_pth.with_name("cprofile.pstats"))
                        if hasattr(mod, "n_iter_"):
                            LOGGER.debug(
                                f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k})/{reg_meth} : model converged in {mod.n_iter_} iterations"
                            )
                        LOGGER.info(
                            f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k})/{reg_meth} : saving scores to {out_pth}"
                        )
                        pd.DataFrame({"Y_hat": Y}).set_index(batch.index).to_csv(out_pth, index_label="cell", sep="\t")


@dataclass(frozen=True, kw_only=True)
class RunBinConf(BaseConf):
    type BinDict = dict[
        tuple[str, str | None],
        tuple[
            Callable[[], DataTup],
            dict[tuple[str, int], dict[str, dict[str, BinFn]]],
        ],
    ]
    bin_runs: BinDict

    @staticmethod
    def mk_dict(cache: Path, bin_runs: dict[str, dict[tuple[str, int], dict[str, list[str]]]]) -> BinDict:

        reg_dict = RunRegConf.mk_dict(
            cache, {ds: {emb: list(regs.keys()) for emb, regs in bins.items()} for ds, bins in bin_runs.items()}
        )

        bin_fns: dict[str, BinFn] = dict()

        for run in bin_runs.values():
            for regs in run.values():
                for bins in regs.values():
                    for bin_meth in bins:
                        if bin_meth not in BIN_CONF:
                            msg = f"{bin_meth} : binarization method must be in {list(BIN_CONF.keys())} but got {bin_meth}"
                            LOGGER.critical(msg)
                            raise ValueError(msg)
                        bin_fns[bin_meth] = BIN_CONF[bin_meth]

        return {
            (ds, grp): (
                fn,
                {
                    (emb, k): {
                        rnm: {bnm: bin_fns[bnm] for bnm in bin_runs[f"{ds}/{grp}" if grp is not None else ds][(emb, k)][rnm]}
                        for rnm in regs.keys()
                    }
                    for (emb, k), regs in ec.items()
                },
            )
            for (ds, grp), (fn, ec) in reg_dict.items()
        }

    @classmethod
    def from_args(cls, args: RunBinArgs) -> Self:
        LOGGER.debug(f"{cls.__name__}.from_args : got args - {args}")

        return cls(
            bin_runs=cls.mk_dict(
                args.cache,
                {
                    ds: {emb: {reg: args.bin_methods for reg in args.reg_methods} for emb in args.emb_methods}
                    for ds in args.datasets
                },
            ),
            cache=args.cache,
            skip=args.skip_if_exists,
            catch=args.catch_on_err,
            seed=args.seed,
        )

    def run(self) -> None:
        err_log_fn = LOGGER.error if self.catch else LOGGER.critical

        for (dataset, grp), (dfn, embs) in self.bin_runs.items():
            _, V, batch = dfn()
            for (emb_meth, k), regs in embs.items():
                for reg_meth, bins in regs.items():
                    df = pd.read_csv(self.cache / mk_reg_path(dataset, grp, emb_meth, k, reg_meth), sep="\t", index_col="cell")
                    if np.any(df.index != batch.index):
                        msg = f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k}) : regression output index does not match dataset index"
                        LOGGER.critical(msg)
                        raise ValueError(msg)
                    Y = df["Y_hat"].to_numpy()

                    for bin_meth, bfn in bins.items():
                        out_pth = self.cache / mk_bin_path(dataset, grp, emb_meth, k, reg_meth, bin_meth)

                        if out_pth.exists():
                            if self.skip:
                                LOGGER.info(
                                    f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k})/{reg_meth}/{bin_meth} : skipping because output file {out_pth} already exists"
                                )
                                continue
                            LOGGER.warning(
                                f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k})/{reg_meth}/{bin_meth} : output file {out_pth} already exists, will be overwritten"
                            )

                        out_pth.parent.mkdir(parents=True, exist_ok=True)

                        LOGGER.info(
                            f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k})/{reg_meth}/{bin_meth} : running binarization"
                        )
                        prof = Profile()
                        try:
                            with Tracker(destination=FileDestination(out_pth.with_name("memray.bin"), overwrite=True)):
                                W = prof.runcall(bfn, Y, V)
                        except Exception as exc:
                            err_log_fn(
                                f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k})/{reg_meth}/{bin_meth} : caught exception during binarization, see below for traceback\n{'\n'.join(format_exception(exc))}\n"
                            )
                            if not self.catch:
                                raise exc
                        else:
                            prof.dump_stats(out_pth.with_name("cprofile.pstats"))
                            LOGGER.info(
                                f"{dataset}{'' if grp is None else f'/{grp}'}/{emb_meth}(k={k})/{reg_meth}/{bin_meth} : saving binarized labels to {out_pth}"
                            )
                            pd.DataFrame({"V_hat": W}).set_index(batch.index).to_csv(out_pth, index_label="cell", sep="\t")


def run_fcf(cmd_file: Path, base_args: dict) -> None:
    with open(cmd_file, "r") as f:
        lines = f.readlines()

    emb_dict: dict[str, list[tuple[str, int]]] = dict()
    reg_dict: dict[str, dict[tuple[str, int], list[str]]] = dict()
    bin_dict: dict[str, dict[tuple[str, int], dict[str, list[str]]]] = dict()

    def parse_err(err_str: str, idx: int, line: str) -> Never:
        msg = f"fcf : got error trying to parse line {idx} (`{line}`) of file {cmd_file} - {err_str}"
        LOGGER.critical(msg)
        raise ValueError(msg)

    for idx, line in enumerate(lines):
        cmd = line.strip().split(" ")
        if len(cmd) < 4:
            parse_err("incorrect number of arguments, must be at least 3", idx, line)
        pfx, ds, emb, k, *rem = cmd
        try:
            k = int(k)
        except ValueError:
            parse_err("third entry (k) could not be parsed to integer", idx, line)
        if pfx == "emb":
            if len(rem) != 0:
                parse_err("incorrect number of arguments for `emb` command", idx, line)
            if ds in emb_dict:
                emb_dict[ds].append((emb, k))
            else:
                emb_dict[ds] = [(emb, k)]
        elif pfx == "reg":
            if len(rem) != 1:
                parse_err("incorrect number of arguments for `reg` command", idx, line)
            reg = rem[0]
            if ds in reg_dict:
                if (emb, k) in reg_dict[ds]:
                    reg_dict[ds][(emb, k)].append(reg)
                else:
                    reg_dict[ds][(emb, k)] = [reg]
            else:
                reg_dict[ds] = {(emb, k): [reg]}
        elif pfx == "bin":
            if len(rem) != 2:
                parse_err("incorrect number of arguments for `reg` command", idx, line)
            reg, bin_nm = rem
            if ds in bin_dict:
                if (emb, k) in bin_dict[ds]:
                    if reg in bin_dict[ds][(emb, k)]:
                        bin_dict[ds][(emb, k)][reg].append(bin_nm)
                    else:
                        bin_dict[ds][(emb, k)][reg] = [bin_nm]
                else:
                    bin_dict[ds][(emb, k)] = {reg: [bin_nm]}
            else:
                bin_dict[ds] = {(emb, k): {reg: [bin_nm]}}
        else:
            parse_err(f"incorrect command prefix `{pfx}`, must be in {['emb', 'reg', 'bin']}", idx, line)

    emb_conf = RunEmbConf(**base_args, emb_runs=RunEmbConf.mk_dict(base_args["cache"], emb_dict))
    reg_conf = RunRegConf(**base_args, reg_runs=RunRegConf.mk_dict(base_args["cache"], reg_dict))
    bin_conf = RunBinConf(**base_args, bin_runs=RunBinConf.mk_dict(base_args["cache"], bin_dict))

    for d, c in [(emb_dict, emb_conf), (reg_dict, reg_conf), (bin_dict, bin_conf)]:
        if len(d) > 0:
            c.run()


if __name__ == "__main__":
    assert CMD is not None

    sub_cmd = CMD.cmd
    match sub_cmd:
        case FromCmdFileArgs():
            run_fcf(
                sub_cmd.cmd_file,
                {"cache": sub_cmd.cache, "skip": sub_cmd.skip_if_exists, "catch": sub_cmd.catch_on_err, "seed": sub_cmd.seed},
            )

        case RunEmbArgs():
            RunEmbConf.from_args(sub_cmd).run()

        case RunRegArgs():
            RunRegConf.from_args(sub_cmd).run()

        case RunBinArgs():
            RunBinConf.from_args(sub_cmd).run()

        case ListDatasets():
            print("\n".join(DATA_CONF.keys()))
        case QueryDataset(dataset, cache, min_n):
            if dataset not in DATA_CONF:
                msg = f"{dataset} : dataset must be in {list(DATA_CONF.keys())}, but got {dataset}"
                LOGGER.critical(msg)
                raise ValueError(msg)
            ad_pth, pp_f, V_f, grps_f, _ = DATA_CONF[dataset]
            adata = ad.read_h5ad(cache / "data" / f"{ad_pth}.h5ad", backed="r")
            assert isinstance(adata.obs, pd.DataFrame)
            meta = adata.obs.loc[pp_f(adata.obs)]
            V, grps = V_f(meta), grps_f(meta)
            print(
                "\n".join(
                    str(grp)
                    for grp, idx in grps.groupby(grps, observed=True).indices.items()
                    if (len(idx) >= min_n) and (len(np.unique(V[idx])) > 1)
                )
            )

        case ListEmbs(cachable):
            if cachable:
                print("\n".join(k for k, (_, b) in EMB_NAIVE_CONF.items() if b))
            else:
                print("\n".join([*(k for k, (_, b) in EMB_NAIVE_CONF.items() if not b), *EMB_AWARE_CONF.keys()]))

        case ListRegs():
            print("\n".join(REG_CONF.keys()))

        case ListBins():
            print("\n".join(BIN_CONF.keys()))
