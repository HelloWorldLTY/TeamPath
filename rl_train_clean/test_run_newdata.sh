#!/bin/bash
#SBATCH --mem=250g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=35
#SBATCH --partition=gpu_h200
#SBATCH --time=24:00:00
#SBATCH --job-name=pytorch
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=verbose,closest

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export WANDB_API_KEY=9e4cf6fe3f70aea31153d9af82f5cbefac3cfaf5
export HF_TOKEN=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

DATA_DIR="8chunks_correctpathmmu" 
LOG_SAVE_DIR="/nfs/roberts/scratch/pi_hz27/tl688/ckpt_pathor1_1e6_data_save60"
PROJECT_NAME="testpath"
EXP_NAME="pathtest1"
VERL_ROOT="./"


python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=["$DATA_DIR/path_reason_mcqa_train.0.parquet","$DATA_DIR/path_reason_mcqa_train.1.parquet","$DATA_DIR/path_reason_mcqa_train.2.parquet","$DATA_DIR/path_reason_mcqa_train.3.parquet","$DATA_DIR/path_reason_mcqa_train.4.parquet","$DATA_DIR/path_reason_mcqa_train.5.parquet","$DATA_DIR/path_reason_mcqa_train.6.parquet","$DATA_DIR/path_reason_mcqa_train.7.parquet"] \
    data.val_files="$DATA_DIR/path_reason_mcqa_train.127.parquet" \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images \
    actor_rollout_ref.model.path=WenchuanZhang/Patho-R1-7B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-06 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    +actor_rollout_ref.rollout.limit_images=1 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.watchdog_timeout=1800 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests=16 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=4\
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=-1 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="$LOG_SAVE_DIR/checkpoints" \
    trainer.validation_data_dir="$LOG_SAVE_DIR/validation_data" \
    trainer.rollout_data_dir="$LOG_SAVE_DIR/rollout_data" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    custom_reward_function.path="$VERL_ROOT/verl/utils/reward_score/path.py" \
    custom_reward_function.name=compute_score \
    hydra.run.dir="$LOG_SAVE_DIR/outputs" \
    hydra.sweep.dir="$LOG_SAVE_DIR/multirun"