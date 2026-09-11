from transformers import PretrainedConfig

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
