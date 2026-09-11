import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

from models.neuroworld import NeuroWorldLM

class HokieConfig(PretrainedConfig):
    model_type = "hokie_lm"
    def __init__(
        self,
        vocab_size=50257,
        d_model=256,
        d_state=16,
        num_categoricals=8,
        num_classes=8,
        num_layers=4,
        max_rollout_steps=4,
        num_branches=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.max_rollout_steps = max_rollout_steps
        self.num_branches = num_branches

class HokieLMForCausalLM(PreTrainedModel):
    config_class = HokieConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = NeuroWorldLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_state=config.d_state,
            num_categoricals=config.num_categoricals,
            num_classes=config.num_classes,
            num_layers=config.num_layers,
            max_rollout_steps=config.max_rollout_steps,
            num_branches=config.num_branches
        )

    def forward(self, input_ids, labels=None, **kwargs):
        logits, loss, metrics, _ = self.model(input_ids, targets=labels, use_posterior=False)
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def generate(self, input_ids, max_new_tokens=40, temperature=0.7, **kwargs):
        gen_tokens = self.model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        return torch.cat([input_ids, gen_tokens], dim=-1)
