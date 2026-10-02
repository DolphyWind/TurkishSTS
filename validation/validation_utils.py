from typing import Dict, List, Set
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
import torch


MODEL_NAMES: List[str] = [
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "ytu-ce-cosmos/turkish-e5-large",
    "ytu-ce-cosmos/modernbert-tr-embed",
    "atasoglu/turkish-e5-large-m2v",
]


def shorten_name(
    s: str,
    max_len: int = 25,
) -> str:
    if len(s) < max_len:
        return s

    half: int = int((max_len - 3) // 2)
    return s[:half] + "..." + s[-half:]


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


def evaluate_model(
    sentences1: List[str],
    sentences2: List[str],
    scores: List[float],
    model_name: str,
) -> Dict[str, float]:
    print(f"Processing: {model_name}")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model: SentenceTransformer = SentenceTransformer(
        model_name,
        device=device,
    )
    evaluator: EmbeddingSimilarityEvaluator = EmbeddingSimilarityEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        show_progress_bar=True,
        similarity_fn_names=['cosine', 'dot', 'euclidean', 'manhattan'],
    )

    return evaluator(model)


def evaluate_all_and_print(
    sentences1: List[str],
    sentences2: List[str],
    scores: List[float],
    max_model_len: int,
) -> None:
    results: Dict[str, Dict[str, float]] = dict()

    for model_name in MODEL_NAMES:
        results[model_name] = evaluate_model(
            sentences1=sentences1,
            sentences2=sentences2,
            scores=scores,
            model_name=model_name,
        )

    print_results_table(
        results,
        max_model_len,
    )
