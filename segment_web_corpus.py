import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from wtpsplit import SaT

BASE_PATH: Path = Path("/run/media/yunus/HDD_5TB/data/NLP/vngrs-web-corpus/data/")
STATE_NAME: str = "state.json"


def extract_sentences_sat(
    sat: SaT,
    texts: List[str],
    batch_size: int,
    stride: int,
) -> List[List[str]]:
    return [
        [s.strip() for s in sentences if s.strip()]
        for sentences in sat.split(texts, batch_size=batch_size, stride=stride)
    ]


def load_state(output_dir: Path) -> Tuple[str, int]:
    state_path: Path = output_dir / STATE_NAME
    if not state_path.exists():
        return "", 0
    with open(state_path, "r", encoding="utf-8") as handle:
        state: Dict[str, Any] = json.load(handle)
    return str(state["file"]), int(state["row"])


def save_state(
    output_dir: Path,
    file_name: str,
    row: int,
) -> None:
    state: Dict[str, Any] = {"file": file_name, "row": row}
    with open(output_dir / STATE_NAME, "w", encoding="utf-8") as handle:
        json.dump(state, handle)


def load_partial(output_path: Path) -> Tuple[List[str], List[str], List[str]]:
    if not output_path.exists():
        return [], [], []
    df: pd.DataFrame = pd.read_parquet(output_path)
    return (
        df["sentence"].tolist(),
        df["source"].tolist(),
        df["category"].tolist(),
    )


def write_output(
    output_path: Path,
    sentences: List[str],
    sources: List[str],
    categories: List[str],
) -> None:
    df: pd.DataFrame = pd.DataFrame({
        'sentence': sentences,
        'source': sources,
        'category': categories,
    })
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        nargs="?",
        default=str(BASE_PATH),
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=str(Path(__file__).parent / "parquet"),
    )
    parser.add_argument(
        "--sat_model",
        type=str,
        choices=[
            "sat-12l-no-limited-lookahead",
            "sat-9l-no-limited-lookahead",
            "sat-6l-no-limited-lookahead",
            "sat-3l-no-limited-lookahead",
            "sat-1l-no-limited-lookahead",
            "sat-12l",
            "sat-9l",
            "sat-6l",
            "sat-3l",
            "sat-1l",
            "sat-12l-sm",
            "sat-9l-sm",
            "sat-6l-sm",
            "sat-3l-sm",
            "sat-1l-sm",
        ],
        default="sat-12l-sm",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--print-sentences",
        action='store_true',
    )
    parser.add_argument(
        "--no-save",
        action='store_true',
    )

    return parser.parse_args()


def main():
    args: argparse.Namespace = parse_args()

    sat: SaT = SaT(args.sat_model)
    sat.half().to('cuda')

    input_dir: Path = Path(args.input_dir)
    parquet_files: List[Path] = sorted(input_dir.glob("*.parquet"))
    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file, state_row = load_state(output_dir)
    TMP_PATH: Path = Path("/tmp")

    for parquet_path in parquet_files:
        if parquet_path.name < state_file:
            continue
        output_path: Path = output_dir / f"{parquet_path.stem}_sat.parquet"

        # I'm doing this because the files sit on my HDD.
        open_path: Path = Path("/tmp") / parquet_path.name
        if not open_path.exists():
            parquet_path.copy_into(TMP_PATH)

        source_df: pd.DataFrame = pd.read_parquet(open_path)
        start_row: int = state_row if parquet_path.name == state_file else 0
        if start_row >= len(source_df):
            continue

        all_sentences: List[str] = []
        all_sources: List[str] = []
        all_categories: List[str] = []
        if start_row > 0:
            all_sentences, all_sources, all_categories = load_partial(output_path)
        total_sentences: int = len(all_sentences)

        loop: tqdm = tqdm(total=len(source_df), initial=start_row)
        loop.set_description_str(parquet_path.name)
        row_index: int = start_row

        try:
            while row_index < len(source_df):
                chunk: pd.DataFrame = source_df.iloc[row_index:row_index + args.chunk_size]
                texts: List[str] = chunk["text"].astype(str).tolist()
                sources: List[str] = chunk["original_id"].astype(str).tolist()
                categories: List[str] = chunk["corpus"].astype(str).tolist()
                results: List[List[str]] = extract_sentences_sat(
                    sat, texts, args.batch_size, args.stride
                )

                for sentences, source, category in zip(results, sources, categories):
                    all_sentences += sentences
                    all_sources += [source] * sentences.__len__()
                    all_categories += [category] * sentences.__len__()
                    total_sentences += sentences.__len__()

                row_index += len(chunk)
                loop.update(len(chunk))
                loop.set_postfix(sentences=total_sentences)
        except KeyboardInterrupt:
            loop.close()
            if not args.no_save:
                write_output(output_path, all_sentences, all_sources, all_categories)
            save_state(output_dir, parquet_path.name, row_index)
            print(f"Interrupted at {parquet_path.name} row {row_index}, state saved")
            return

        if open_path.exists():
            os.remove(open_path.__str__())
        if args.print_sentences:
            for s in all_sentences:
                print(f"SENTENCE: `{s}`")

        if not args.no_save:
            write_output(output_path, all_sentences, all_sources, all_categories)
        save_state(output_dir, parquet_path.name, len(source_df))


if __name__ == '__main__':
    main()
