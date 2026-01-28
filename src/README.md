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

---

## Evaluating with URDF Objects

You can evaluate pi0.5 with URDF objects from the `sim-evals/objects/` folder. This mode loads a minimal scene with just the table and your specified object.

### Available Objects

Objects are stored in `sim-evals/objects/`:
- `103275_evogen` - Stapler (Evogen-refined URDF)
- `103275_ivw` - Stapler (IVW-refined URDF)
- `103275_humangensim2` - Stapler (Human GenSim2 URDF)
- `103275_vlm1shot` - Stapler (VLM 1-shot URDF)

### Usage

```bash
# Basic usage - load an object by name
python run_eval.py --object 103275_evogen --instruction "press the stapler"

# With custom position and rotation
python run_eval.py --object 103275_evogen \
  --object-pos 0.5 0.0 0.065 \
  --object-rot 1.0 0.0 0.0 0.0 \
  --instruction "press the stapler"

# Full example with all options
python run_eval.py \
  --object 103275_evogen \
  --episodes 10 \
  --headless \
  --client jointvel \
  --instruction "press the stapler" \
  --object-pos 0.5 0.0 0.065

# Interactive mode (show viewer)
python run_eval.py --object 103275_evogen --instruction "press the stapler" --no-headless
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--object` | Object folder name in `sim-evals/objects/` | None (use scene mode) |
| `--object-pos` | Object position (x, y, z) | (0.5, 0.0, 0.065) |
| `--object-rot` | Object rotation quaternion (w, x, y, z) | (1.0, 0.0, 0.0, 0.0) |
| `--instruction` | Task instruction for the policy | Auto-detected |
| `--episodes` | Number of evaluation episodes | 10 |
| `--headless` | Run without GUI | True |
| `--client` | Control mode: `jointvel` or `jointpos` | `jointvel` |

### Adding New Objects

To add a new object for evaluation:

1. Create a folder in `sim-evals/objects/` with your object name (e.g., `my_object`)
2. Add a URDF file: `mobility_revised.urdf` (preferred) or `mobility.urdf`
3. Include mesh files referenced by the URDF (typically in `textured_objs/` subfolder)
4. Run: `python run_eval.py --object my_object --instruction "your instruction"`
