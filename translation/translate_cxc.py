import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List
import warnings

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as hf_logging
import torch
from tqdm import tqdm


PROMPT: str = (
    "Translate this from English to Turkish:\n"
    "English: SENTENCE_HERE\nTurkish:"
)


@torch.no_grad()
def main():
    hf_logging.set_verbosity_error()
    warnings.filterwarnings(
        "ignore",
        message=r".*MatMul8bitLt",
    )
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
    model_id = "xiaomi-research/MiLMMT-46-4B-v1.0"
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir="/run/media/yunus/HDD_5TB/hf_cache",
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "6GiB", "cpu": "64GiB"},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        cache_dir="/run/media/yunus/HDD_5TB/hf_cache",
    )

    agg_dataset: DatasetDict = load_dataset("DolphyWind/cxc-sts-sentences", "aggregated")

    data: Dict[str, List[Dict[str, Any]]] = dict()

    for split in agg_dataset.keys():
        dataset = agg_dataset[split]
        data[split] = []
        for row in tqdm(dataset):
            s1: str = row["caption1"]
            s2: str = row["caption2"]
            score: float = row["agg_score"]
            sampling_method: str = row["sampling_method"]
            text1 = PROMPT.replace("SENTENCE_HERE", s1)
            text2 = PROMPT.replace("SENTENCE_HERE", s2)
            inputs = tokenizer(
                [text1, text2],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                return_tensors="pt",
                padding_side="left",
            ).to("cuda")

            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                output_logits=True,
                return_dict_in_generate=True,
            )
            for step, logits in enumerate(outputs.logits):
                if torch.isnan(logits).any():
                    print(f"NaN in raw logits at step {step}", file=sys.stderr)
                    exit(1)
                if torch.isinf(logits).any():
                    print(f"Inf in raw logits at step {step}", file=sys.stderr)
                    exit(1)
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = outputs.sequences[:, prompt_len:]

            out1, out2 = tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )
            data[split].append({
                "caption1": out1,
                "caption2": out2,
                "agg_score": score,
                "sampling_method": sampling_method,
            })

    data_folder: Path = Path("data/")
    data_folder.mkdir(
        exist_ok=True,
        parents=True,
    )

    for split, rows in data.items():
        deserialized: List[str] = list(map(
            lambda x: json.dumps(
                x,
                ensure_ascii=False,
            ),
            rows,
        ))
        out: str = "\n".join(deserialized)

        with open(
            data_folder / f"cxc-{split}.jsonl",
            "w",
        ) as f:
            f.write(out)


if __name__ == '__main__':
    main()
