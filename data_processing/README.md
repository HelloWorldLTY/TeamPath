# Data preprocessing

This folder contains the TeamPath data construction notebook, `tutorial.ipynb`. It covers ROI-level pathology QA data, ROI testing data, and spatial transcriptomics data used by the SFT and RL pipelines.

## Folder role

- `tutorial.ipynb`: end-to-end examples for turning pathology datasets into training files.
- ROI RL output: verl-compatible parquet shards, usually named like `path_reason_mcqa_train.0.parquet`.
- Spatial transcriptomics output: LLaMA-Factory style JSON files such as `gliomas_train.json`.

## PathReason ROI data for RL

The first notebook section processes PathGen/PathReason-style closed-choice pathology QA data into parquet shards for `rl_train_clean`.

Input assumptions:

- Download PathGen 1.6M from Hugging Face: `https://huggingface.co/datasets/jamessyx/PathGen`.
- The notebook expects a PathReason instruction file like `PathReason/PathGen-Instruct.json`.
- ROI image paths are resolved relative to the data root and converted to RGB PIL images.

Output behavior:

- Keeps only records where `type == "CLOSE"`.
- Builds one sample per human/GPT QA pair.
- Prepends `<image>` to the user question and removes duplicate image markers.
- Requires the ground-truth answer to be a single capital letter.
- Splits the processed dataset across `chunk_num` parquet shards.
- Saves occasional debug images next to the generated parquet files.

Each RL parquet row follows this schema:

```json
{
  "data_source": "path_reason",
  "prompt": [
    {
      "role": "user",
      "content": "<image>\n... Think in <think>...</think> and answer in <answer>...</answer>."
    }
  ],
  "images": ["..."],
  "ability": "pathology",
  "reward_model": {
    "style": "rule",
    "ground_truth": "A"
  }
}
```

The RL dataloader reads `prompt` as chat messages and uses `images` through `data.image_key=images`. The TeamPath reward function in `rl_train_clean/verl/utils/reward_score/path.py` extracts the final `<answer>...</answer>` span and scores MCQA answers by matching the single capital-letter ground truth.

## Spatial transcriptomics data

The later notebook sections prepare spatial transcriptomics instruction data from HEST-1K and STimage-1K4M style inputs.

Input assumptions:

- Load processed `.h5ad` files with `scanpy`.
- Generate each spot's top expressed genes by ranking expression values.
- Save intermediate CSVs under `image_info`.

Output behavior:

- Creates instruction/answer pairs asking the model to generate the top `num_genes` genes.
- Builds LLaMA-Factory multimodal JSON with `messages` and `images`.
- Writes JSON such as `gliomas_train.json`.

## Hand-off to training

For RL training:

1. Generate PathReason parquet shards with the notebook or an equivalent script.
2. Put all shards in one `DATA_DIR`.
3. Point `rl_train_clean/test_run_newdata.sh` at that directory.
4. Keep `custom_reward_function.path` set to `verl/utils/reward_score/path.py`.
5. Keep `data.image_key=images` unless you change the parquet column name.

For SFT:

1. Generate LLaMA-Factory compatible JSON files with `messages` and `images`.
2. Put image paths where the training config can resolve them.
3. Follow the SFT README and the project training script referenced from the top-level README.
