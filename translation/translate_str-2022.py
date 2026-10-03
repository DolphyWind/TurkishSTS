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
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        cache_dir="/run/media/yunus/HDD_5TB/hf_cache",
    )
    dataset: Dataset = load_dataset("vkpriya/str-2022")["train"]

    data: List[Dict[str, Any]] = []

    for row in tqdm(dataset):
        sentences = row["Text"]
        s1, s2 = sentences.split("\n")
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

        row["sentence1"] = out1
        row["sentence2"] = out2
        row["Text"] = f"{out1}\n{out2}"
        data.append(row)

    deserialized: List[str] = list(map(
        lambda x: json.dumps(x, ensure_ascii=False),
        data
    ))
    out: str = "\n".join(deserialized)

    data_folder: Path = Path("data/")
    data_folder.mkdir(exist_ok=True, parents=True)
    with open(data_folder / "str-2022-train.jsonl", "w") as f:
        f.write(out)


if __name__ == '__main__':
    main()
