ckpt_dir="/nfs/roberts/scratch/pi_hz27/tl688/ckpt_pathor1_1e6_pathinstructpathmmu_save400/checkpoints/"

for step in {100..1691..100}
do
    echo $ckpt_dir/global_step_${step}
    which python
    python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir $ckpt_dir/global_step_${step}/actor \
    --target_dir $ckpt_dir/global_step_${step}/actor/merged_hf_model
done