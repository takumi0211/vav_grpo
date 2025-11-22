import random
from torch.utils.data import IterableDataset
import torch


class StepStream(IterableDataset):
    """Stream prompts and reward columns for GRPO training.

    Each iteration samples ``k`` distinct prompts from the base dataset and
    yields ``num_generations`` copies of the prompt with reward tensors so that
    TRL can draw multiple completions per prompt.
    """

    KEEP_KEYS = (
        "reward_action_0",
        "reward_action_1",
        "reward_action_2",
        "reward_action_3",
        "prompt",
    )

    def __init__(self, base_ds, k, num_generations):
        self.base = base_ds
        self.n = len(base_ds)
        if self.n == 0:
            raise ValueError("StepStream requires a non-empty dataset.")
        self.k = min(k, self.n)
        self.num_generations = num_generations
        # Use only columns that actually exist in the dataset.
        dense_keys = [key for key in self.KEEP_KEYS if key in base_ds.features and key != "prompt"]
        if "prompt" in base_ds.features:
            dense_keys.append("prompt")
        self.keys = dense_keys

    def __iter__(self):
        while True:
            idxs = random.sample(range(self.n), self.k)
            for i in idxs:
                row = self.base[i]
                sample = {}
                for key in self.keys:
                    value = row[key]
                    if key == "prompt":
                        sample[key] = value
                    else:
                        sample[key] = torch.atleast_1d(
                            torch.tensor(value, dtype=torch.float32)
                        )
                for _ in range(self.num_generations):
                    yield {
                        key: (value.clone() if isinstance(value, torch.Tensor) else value)
                        for key, value in sample.items()
                    }
