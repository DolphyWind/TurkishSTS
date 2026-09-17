from typing import Any, Dict, Optional, Union

from peft import LoraConfig, get_peft_model, PeftModel
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class STSModel(nn.Module):
    def __init__(
        self,
        base_model: str,
        initialize_new: bool = False,
        lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.base_model_name: str = base_model
        self.initialize_new: bool = initialize_new
        self.use_lora: bool = lora
        self.lora_config: Optional[LoraConfig] = LoraConfig(**lora_config) if lora_config else None

        self._base_model: PreTrainedModel
        self.model_config: PreTrainedConfig = AutoConfig.from_pretrained(self.base_model_name)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.initialize_new:
            self._base_model = AutoModel.from_config(self.model_config)
        else:
            self._base_model = AutoModel.from_pretrained(self.base_model_name)

        self.base: Union[PeftModel, PreTrainedModel]
        if self.use_lora:
            assert self.lora_config is not None
            self.base = get_peft_model(self._base_model, self.lora_config)
        else:
            self.base = self._base_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        return self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )


def main():
    pass


if __name__ == '__main__':
    main()
