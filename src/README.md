# Sim-Evals Evaluation

## Running Evaluations

### 1. Start Policy Server

**Original pi05 DROID model:**
```bash
cd /home/huluwulu/Projects/eurekaworld/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/serve_policy.py --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

**Finetuned pi05 model:**
```bash
cd /home/huluwulu/Projects/eurekaworld/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/serve_policy.py --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_droid_lora_finetune \
  --policy.dir=/home/huluwulu/Projects/eurekaworld/openpi/checkpoints/pi05_droid_lora_finetune/eureka_droid_lift_20260105_164339_lora/19999
```

### 2. Run Evaluation

In a separate terminal:
```bash
conda activate env_isaaclab_5.x
cd /home/huluwulu/Projects/eurekaworld/sim-evals
python run_eval.py --episodes 10 --scene 1 --headless --client jointvel --instruction "lift the cube"
```

Videos are saved to `runs/YYYY-MM-DD/HH-MM-SS/episode_N.mp4`.

