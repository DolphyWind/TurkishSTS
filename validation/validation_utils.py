import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Set, Union
from mteb import TaskResult
from peft import PeftModel, TaskType
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from config import ModelConfig  # noqa: E402
from model import STSModel  # noqa: E402


BASE_PATH: str = "/run/media/yunus/HDD_5TB/results/TurkishSentenceSimilarity/"


# TODO: Three different dataclasses for;
#  - Custom model (Ones ending with model_best.pth)
#  - Custom model trained with HF trainers and therefore use HF's save method
#  - Models from hf
MODEL_NAMES: List[Union[str, Path]] = [
    Path(f"{BASE_PATH}/turkish_sts/colab_test_2/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/ablation_regular_run/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/ablation_no_dsl/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/ablation_1M_no_dsl/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/ablation_regular_sampler/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/ablation_low_p_value/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/ablation_uncased/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/ablation_simcse/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/ablation_simcse_with_dsl/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/ablation_fix_sampler/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/ablation_fix_sampler_two/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/ablation_emrecan/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/improved_100K_with_dcl_cased/model_best.pth"),
    Path(f"{BASE_PATH}/turkish_sts/improved_100K_without_dcl_cased/model_best.pth"),
    f"{BASE_PATH}/turkish_sts/mixed-anglE/",
    f"{BASE_PATH}/turkish_sts/stsb-nli-anglE/",
    f"{BASE_PATH}/turkish_sts/mixed-cossim-10ep/",
    f"{BASE_PATH}/turkish_sts/stsb-cxc-anglE-10ep/",
    f"{BASE_PATH}/turkish_sts/cxc-anglE-10ep/",
    f"{BASE_PATH}/turkish_sts/simcse_sup_initial/",
    f"{BASE_PATH}/turkish_sts/mixed-anglE-10ep-after-simcse-sup/",
    f"{BASE_PATH}/turkish_sts/web_pairs/",
    f"{BASE_PATH}/turkish_sts/simcse_sup_cached_mnrl/",
    f"{BASE_PATH}/turkish_sts/simcse_sup_cached_mnrl_2/",
    "trmteb/turkish-embedding-model-fine-tuned",
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "ytu-ce-cosmos/turkish-e5-large",
    "ytu-ce-cosmos/modernbert-tr-embed",
    "dbmdz/bert-base-turkish-cased",
    "dbmdz/bert-base-turkish-uncased",
]


def shorten_name(
    s: str,
    max_len: int = 25,
) -> str:
    if len(s) < max_len:
        return s

    # half: int = int((max_len - 3) // 2)
    # return s[:half] + "..." + s[-half:]
    return "..." + s[-max_len+3:]


def print_results_table(
    results: Dict[str, Dict[str, float]],
    max_model_len: int,
) -> None:
    cols: List[str] = list(next(iter(results.values())).keys())
    max_col: int = max(map(len, cols))
    max_model_name_len: int = max(len(shorten_name(name, max_model_len)) for name in results)

    print(" " * max_model_name_len, end='')
    for col in cols:
        print(f"|{col:^{max_col + 2}}", end='')
    print()

    row_len = max_model_len + (3 + max_col) * len(cols)

    for model_name, res in results.items():
        print("-" * row_len)
        sn: str = shorten_name(model_name, max_model_len)
        print(f"{sn:^{max_model_name_len}}", end='')
        for c, v in res.items():
            v = round(v * 100, 2)
            v_str: str = f"{v}%"
            print(f"|{v_str:^{max_col + 2}}", end='')
        print()
    print("-" * row_len)


def load_sentence_transformer(
    model_name: Union[str, Path],
    device: str,
) -> SentenceTransformer:
    model: SentenceTransformer
    if isinstance(model_name, str):
        model = SentenceTransformer(
            model_name,
            device=device,
        )
    else:
        config_path: Path = model_name.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} does not exist.")
        saved: Dict[str, Any] = json.loads(config_path.read_text())
        model_config: ModelConfig = ModelConfig(**saved["model_config"])
        sts_model: STSModel = STSModel(
            base_model_name=model_config.base_model,
            initialize_new=model_config.initialize_new,
            lora=model_config.lora,
            lora_config={
                "r": model_config.lora_r,
                "lora_alpha": model_config.lora_alpha,
                "lora_dropout": model_config.lora_dropout,
                "bias": "none",
                "task_type": TaskType.FEATURE_EXTRACTION,
                "target_modules": model_config.lora_target_modules,
                "modules_to_save": model_config.lora_modules_to_save,
            },
        )
        state_dict: Dict[str, torch.Tensor] = torch.load(
            model_name,
            map_location=device,
        )
        sts_model.load_state_dict(state_dict)
        if isinstance(sts_model.base, PeftModel):
            sts_model.base = sts_model.base.merge_and_unload()

        with tempfile.TemporaryDirectory() as tmp_dir:
            sts_model.base.save_pretrained(tmp_dir)
            sts_model.tokenizer.save_pretrained(tmp_dir)
            transformer: Transformer = Transformer(tmp_dir)
            pooling: Pooling = Pooling(
                transformer.get_embedding_dimension(),
                "mean",
            )
            model = SentenceTransformer(
                modules=[transformer, pooling],
                device=device,
            )

    return model


def evaluate_model(
    sentences1: List[str],
    sentences2: List[str],
    scores: List[float],
    model_name: Union[str, Path],
) -> Dict[str, float]:
    print(f"Processing: {model_name}")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: SentenceTransformer = load_sentence_transformer(
        model_name,
        device,
    )

    evaluator: EmbeddingSimilarityEvaluator = EmbeddingSimilarityEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        show_progress_bar=True,
        similarity_fn_names=['cosine', 'dot', 'euclidean', 'manhattan'],
    )

    return evaluator(model)


def evaluate_mteb_tr(
    model_name: Union[str, Path],
    mteb_out_folder: Path,
) -> List[TaskResult]:
    import mteb
    from mteb import MTEB

    print(f"Processing: {model_name}")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mteb_tr = mteb.get_benchmark("MTEB(Turkish)")
    model: SentenceTransformer = load_sentence_transformer(
        model_name,
        device,
    )
    evaluation = MTEB(tasks=mteb_tr)
    return evaluation.run(model, output_folder=str(mteb_out_folder))


def evaluate_all_mteb_tr(
    max_model_len: int,
) -> None:
    out_base: Path = Path(BASE_PATH)
    raw_out_folder: Path = out_base / "mteb_raw_out"
    raw_out_folder.mkdir(parents=True, exist_ok=True)

    out_folder: Path = out_base / "mteb_out"
    out_folder.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, float]] = dict()

    cache_path: Path = out_folder / "mteb_tr.json"
    cache: Dict[str, Dict[str, float]] = dict()
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    cache_updated: bool = False

    try:
        for model_name in MODEL_NAMES:
            name: str
            if isinstance(model_name, Path):
                name = model_name.parent.name
            else:
                name = model_name

            key: str = str(model_name)
            if key in cache:
                print(f"Loaded from cache: {model_name}")
                results[name] = cache[key]
                continue

            task_results: List[TaskResult] = evaluate_mteb_tr(
                model_name=model_name,
                mteb_out_folder=raw_out_folder / name,
            )
            sorted_results: List[TaskResult] = sorted(
                task_results,
                key=lambda task_result: task_result.task_name,
            )
            cache[key] = {
                task_result.task_name: float(task_result.get_score())
                for task_result in sorted_results
            }
            cache_updated = True
            results[name] = cache[key]
    finally:
        if cache_updated:
            cache_path.write_text(json.dumps(
                cache,
                indent=4,
            ))

    print_results_table(
        results,
        max_model_len,
    )


def evaluate_all_and_print(
    sentences1: List[str],
    sentences2: List[str],
    scores: List[float],
    max_model_len: int,
    dataset_name: str,
) -> None:
    results: Dict[str, Dict[str, float]] = dict()

    cache_path: Path = Path(f"{BASE_PATH}/validation_cache") / f"{dataset_name}.json"
    cache: Dict[str, Dict[str, float]] = dict()
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    cache_updated: bool = False

    try:
        for model_name in MODEL_NAMES:
            name: str
            if isinstance(model_name, Path):
                name = model_name.parent.name
            else:
                name = model_name

            key: str = str(name)
            if key in cache:
                print(f"Loaded from cache: {model_name}")
                results[name] = cache[key]
                continue

            metrics: Dict[str, float] = evaluate_model(
                sentences1=sentences1,
                sentences2=sentences2,
                scores=scores,
                model_name=model_name,
            )
            cache[key] = {k: float(v) for k, v in metrics.items()}
            cache_updated = True
            results[name] = cache[key]
    finally:
        if cache_updated:
            cache_path.parent.mkdir(
                exist_ok=True,
                parents=True,
            )
            cache_path.write_text(json.dumps(
                cache,
                indent=4,
            ))

    print_results_table(
        results,
        max_model_len,
    )
