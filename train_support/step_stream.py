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

    def __init__(self, base_ds, k, num_generations, extra_keys=None):
        self.base = base_ds
        self.n = len(base_ds)
        if self.n == 0:
            raise ValueError("StepStream requires a non-empty dataset.")
        self.k = min(k, self.n)
        self.num_generations = num_generations

        extra_keys = tuple(extra_keys or ())

        # Build key order so that at least one numeric/tensor field appears first.
        reward_keys = [
            key for key in self.KEEP_KEYS if key.startswith("reward") and key in base_ds.features
        ]
        numeric_pref = []
        for key in ("sample_id", "episode_id", "step", "idx"):
            if key in base_ds.features and key not in reward_keys:
                numeric_pref.append(key)

        aux_keys = []
        for key in extra_keys:
            if key == "prompt":
                continue  # force prompt to the end
            if key in base_ds.features and key not in reward_keys and key not in numeric_pref:
                aux_keys.append(key)

        dense_keys = reward_keys + numeric_pref + aux_keys

        # Prompt is kept last so Accelerate's find_batch_size sees a tensor first.
        if "prompt" in base_ds.features:
            dense_keys.append("prompt")

        # Fallback: if nothing tensor-like is available, inject a dummy key.
        if not dense_keys or dense_keys[0] == "prompt":
            dense_keys.insert(0, "_dummy_batch_size")

        self.keys = dense_keys

    def __iter__(self):
        while True:
            idxs = random.sample(range(self.n), self.k)
            for i in idxs:
                row = self.base[i]
                sample = {}
                for key in self.keys:
                    if key == "_dummy_batch_size":
                        sample[key] = torch.tensor([0.0], dtype=torch.float32)
                        continue

                    value = row[key]
                    if key == "prompt":
                        sample[key] = value
                    elif isinstance(value, (str, bytes)):
                        # Non-numeric metadata (state_json, sample_id string, etc.)
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
