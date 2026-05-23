# Dataset Format
## RLHF dataset
We combine all the data sources into a single parquet files. We directly organize the prompt into the chat format so that multi-turn chats can be easily incorporated. In the prompt, we may add instruction following texts to guide the model output the answers in a particular format so that we can extract the answers.

TeamPath pathology RL data uses the same parquet contract with an additional multimodal `images` column. The training config should set `data.image_key=images`, and the user prompt should include an `<image>` marker so the processor inserts the visual input into the chat template.

Pathology MCQA example
```json
{
    "data_source": "path_reason",
    "prompt": [{"role": "user", "content": "<image>\nWhat is shown in this pathology ROI?\n(A) ...\n(B) ...\nYour task: think in <think>...</think> and answer in <answer>...</answer>."}],
    "images": ["path/to/roi.png"],
    "ability": "pathology",
    "reward_model": {
        "style": "rule",
        "ground_truth": "A"
    }
}
```

The TeamPath reward function at `verl/utils/reward_score/path.py` expects `reward_model.ground_truth` to contain either a single capital-letter MCQA answer or a longer text target. Single-letter targets are scored by exact answer matching; longer targets are scored with BLEU.

Math problems
```json
{
    "data_source": "openai/gsm8k",
    "prompt": [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\""}],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": ["72"]
    },
}
```
