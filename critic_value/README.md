# `critic_value/` – Using the TD3 critic as a GRPO reward model

This package wraps a TD3 critic trained for the multi‑zone HVAC simulator and
exposes utilities to reuse it as a **reward model** when training an LLM with
GRPO (or any RLHF‑style algorithm that needs scalar rewards for actions).

At a high level:

- The **TD3 checkpoint** (`td3_policy_final.pt`) contains:
  - the environment/config (zone count, actuator ranges, comfort/CO₂ targets),
  - the `ObservationNormalizer` used during RL training,
  - the twin Q‑network weights (`TwinQNetwork`).
- `critic_value/` scripts turn logs or decision CSVs into:
  - `critic_value/data/dataset.csv` – prompts and normalized states,
  - `critic_value/data/value.csv` – Q‑values for (state, action) pairs.
- These Q‑values can be mapped to GRPO rewards to prefer higher‑value actions.

The sections below focus on how to make this directory act as a **reward
model service** for an LLM.

---

## 1. Components

- `td3_policy_final.pt`
  - TD3 checkpoint trained in the HVAC simulator.
  - Provides the **critic network** and **observation normalizer**.

- `build_llm_prompt_dataset.py`
  - Rebuilds `critic_value/data/dataset.csv` from an LLM actions log
    (e.g. `outputs/llm_gpt5/llm_actions_log.csv`).
  - For each log row it:
    - parses the observation JSON embedded in the original prompt,
    - reconstructs the raw state vector in the TD3 observation order,
    - normalizes the state using the checkpoint’s `ObservationNormalizer`,
    - stores:
      - `sample_id`
      - `timestamp`
      - `state_raw_json` (unnormalized observation)
      - `state_json` (normalized observation)
      - `prompt` (clean, deterministic prompt text to feed the LLM).

- `build_grpo_dataset.py`
  - Builds a **decision dataset with action variants** from TD3 rollout
    decision CSVs.
  - For each selected decision step it:
    - creates a “base” action (the action actually taken),
    - creates several **action variants** by nudging the tanh‑space actions
      (e.g. `vent_up`, `energy_save`, etc.),
    - writes all variants into `critic_value/data/dataset.csv`, with columns:
      - observation features,
      - `action_*_tanh` columns,
      - `variant_id`, `variant_name`,
      - `sample_id`.

- `evaluate_critic.py`
  - Runs the TD3 critic over a CSV of [obs, action] rows.
  - Reads `critic_value/data/dataset.csv` (or another CSV you pass by
    `--dataset`) and writes `critic_value/data/value.csv` with:
    - `sample_id`
    - `q1`, `q2`
    - `q_min` (elementwise min of the twin critics)
    - `q_mean` (average of the twin critics)
    - plus any passthrough metadata columns.

- `llm_tools.py`
  - Helpers for **online** evaluation of LLM actions against the critic:
    - `resolve_observation_vector(row, normalizer, obs_dim)` – recovers a
      normalized observation vector from a dataset row, using `state_json`
      and/or `state_raw_json`.
    - `build_action_bounds(config)` – computes low/high actuator bounds from
      the TD3 config (damper, OA, coil, fan).
    - `action_dict_to_vector(action, zone_count, low_bounds, high_bounds)` –
      converts an LLM JSON payload into a clipped action vector in the
      critic’s action order.
    - `evaluate_candidate(critic, scaler, device, obs_tensor_norm, action_vec_scaled)` –
      returns `{"q1", "q2", "q_min", "q_mean"}` for a single (state, action)
      pair.

Together, these pieces let you convert:

1. a simulator observation → normalized TD3 observation vector,
2. an LLM JSON action → TD3 action vector,
3. into one or more Q‑values that act as **rewards** for GRPO.

---

## 2. Offline GRPO dataset: from TD3 rollouts

This flow is useful when you already have TD3 rollouts and want to construct
an offline GRPO dataset where each prompt has multiple candidate actions,
each with a scalar reward derived from the critic.

### 2.1. Build the GRPO decision dataset

Input:

- One or more “decision step” CSVs from the simulator (one per day), typically
  containing:
  - observation features,
  - tanh‑space action columns (e.g. `action_zone_1_tanh`, `action_oa_tanh`),
  - metadata such as `episode_id`, timestamps, etc.

Command (example):

```bash
python critic_value/build_grpo_dataset.py \
  --sources critic_value/data/decisions_20250729.csv \
  --per-day 25 \
  --variants 4 \
  --output critic_value/data/dataset.csv
```

What this does:

- Samples up to `per-day` decision steps per input file (approximately evenly
  spaced in time/episodes).
- For each sampled step, generates `1 + variants` rows:
  - one `variant_name == "base"` row (original TD3 action),
  - several variant rows with slightly perturbed tanh actions.
- Assigns each logical underlying decision a `sample_id`, and each variant a
  `variant_id` and `variant_name`.
- Writes the combined dataset to `critic_value/data/dataset.csv`.

### 2.2. Score the variants with the critic

Once `dataset.csv` contains both observations and `action_*_tanh` columns, run:

```bash
python critic_value/evaluate_critic.py \
  --checkpoint critic_value/td3_policy_final.pt \
  --dataset critic_value/data/dataset.csv \
  --output critic_value/data/value.csv
```

This will:

- Load the TD3 config, normalizer, and critic weights from
  `td3_policy_final.pt`.
- Normalize the observation features as in RL training.
- Evaluate each (obs, action) row with the twin Q‑network.
- Write `critic_value/data/value.csv` with Q‑values and passthrough metadata.

### 2.3. Map critic scores to GRPO rewards

`value.csv` now contains per‑variant scores (`q_min`, `q_mean`, etc.) that can
be used as **rewards** or relative preferences for GRPO‑style training.

Typical choices:

- Use `q_min` as a **pessimistic scalar reward**:
  - more robust when the two critics disagree.
- Use `q_mean` as a **smoothed reward**:
  - higher is better; subtract a baseline if needed.

For each logical decision (grouped by `sample_id`):

1. Collect all variants (distinguished by `variant_id` / `variant_name`).
2. Compute a reward for each variant (e.g. `reward = q_min`).
3. Build GRPO examples with:
   - `prompt` (from your own prompt building logic),
   - `response` (the textual action output that produced this variant),
   - `reward` (float derived from the critic).

The exact GRPO schema depends on the training framework, but the critic’s role
is always to provide **monotonic scores** where “better HVAC control” →
“higher Q‑value” → “higher reward”.

---

## 3. Online reward model: scoring LLM actions on the fly

Instead of precomputing `value.csv`, you can treat the TD3 critic as an
online reward model that scores LLM‑proposed actions during training or
evaluation.

The typical loop looks like:

1. **Prepare the state**:
   - From the simulator or log, build an observation dict matching the
     structure used in `build_llm_prompt_dataset.py` (zones, outdoor, previous
     action, timestamp).
   - Use `critic_value.prompt_utils.compose_prompt_from_agent(...)` to create
     a text prompt for the LLM.

2. **Sample from the LLM**:
   - Call the LLM with the prompt and parse its JSON action payload, e.g.:
     - `zone_dampers`: list of floats per zone,
     - `oa_damper`, `coil_valve`, `fan_speed`.

3. **Convert to critic inputs**:
   - Load the TD3 checkpoint and config (as in `evaluate_critic.py`).
   - Use `build_action_bounds(config)` to get per‑actuator min/max.
   - Use `action_dict_to_vector(...)` to clip and arrange the action into a
     vector.
   - Use `resolve_observation_vector(...)` or the same logic as
     `build_llm_prompt_dataset.py` to obtain a normalized observation vector.

4. **Evaluate the candidate**:

   - Pass the observation tensor and scaled action vector to
     `evaluate_candidate(...)`:

   ```python
   result = evaluate_candidate(
       critic=critic,
       scaler=action_scaler,  # same scaler as TD3 training
       device=device,
       obs_tensor_norm=obs_tensor_norm,  # shape [1, obs_dim]
       action_vec_scaled=action_vec_scaled,  # numpy, shape [action_dim]
   )
   reward = result["q_min"]  # or q_mean
   ```

5. **Feed into GRPO**:
   - Use `reward` as the scalar feedback for the sampled response.
   - Optionally, sample several responses per prompt, score each, and use
     their relative rewards for GRPO’s group‑wise optimization.

In this online setup, `critic_value/` is effectively a **reward model
service** powered by the TD3 critic: given `(prompt, action_json)` it returns
a scalar quality estimate for the action.

---

## 4. Summary

- `td3_policy_final.pt` encapsulates the HVAC control critic trained in the
  simulator.
- `build_grpo_dataset.py` + `evaluate_critic.py` turn TD3 rollouts into
  offline GRPO datasets with critic‑based rewards.
- `build_llm_prompt_dataset.py` + `llm_tools.py` support reconstructing
  prompts and scoring LLM actions online.
- To make this directory a **GRPO reward model**, always:
  - feed **normalized observations** and **properly scaled actions** to the
    critic,
  - use `q_min`/`q_mean` as scalar rewards,
  - group variants per `sample_id` so GRPO can compare alternative responses
    to the same prompt.

