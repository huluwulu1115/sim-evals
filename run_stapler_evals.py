#!/usr/bin/env python3
"""Run stapler evaluations for all variants and tasks.

Usage:
    conda activate env_isaaclab_5.x
    python run_stapler_evals.py
"""

import subprocess
import sys
from datetime import datetime

# Configuration
VARIANTS = [
    "stapler/103275_humangensim2",
    "stapler/103275_ivw",
    "stapler/103275_vlm1shot",
]

TASKS = ["grasping", "tool_use"]

EPISODES = 5


def run_eval(variant: str, task: str, episodes: int = EPISODES) -> tuple[bool, str]:
    """Run a single evaluation and return (success, message)."""
    cmd = [
        sys.executable,
        "run_eval.py",
        "--object", variant,
        "--task", task,
        "--episodes", str(episodes),
        "--headless",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per run (evals can be slow)
        )
        if result.returncode == 0:
            return True, "Success"
        else:
            # Extract last few lines of stderr for error message
            stderr_lines = (result.stderr or result.stdout or "").strip().split('\n')[-5:]
            return False, '\n'.join(stderr_lines)
    except subprocess.TimeoutExpired:
        return False, "Timeout (>30 min)"
    except Exception as e:
        return False, str(e)


def progress_bar(current: int, total: int, width: int = 40, prefix: str = "") -> str:
    """Generate a progress bar string."""
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    percent = 100 * current / total
    return f"{prefix}[{bar}] {current}/{total} ({percent:.0f}%)"


def main():
    # Build list of all runs
    runs = [(v, t) for v in VARIANTS for t in TASKS]
    total = len(runs)
    
    print("=" * 70)
    print("STAPLER EVALUATION SUITE")
    print("=" * 70)
    print(f"Variants: {len(VARIANTS)}")
    print(f"Tasks: {TASKS}")
    print(f"Episodes per run: {EPISODES}")
    print(f"Total runs: {total}")
    print("=" * 70)
    print()
    
    results = []
    start_time = datetime.now()
    
    for i, (variant, task) in enumerate(runs):
        # Show progress
        print(progress_bar(i, total, prefix="Progress: "))
        
        # Show current run
        variant_short = variant.split("/")[-1]
        print(f"\n▶ Running: {variant_short} | {task}")
        print(f"  Command: python run_eval.py --object {variant} --task {task} --episodes {EPISODES}")
        
        run_start = datetime.now()
        success, message = run_eval(variant, task)
        run_duration = datetime.now() - run_start
        
        status = "✓" if success else "✗"
        results.append((variant, task, success, message, run_duration))
        
        print(f"  {status} Completed in {run_duration.total_seconds():.1f}s")
        if not success:
            print(f"  Error: {message[:100]}...")
        print()
    
    # Final progress
    print(progress_bar(total, total, prefix="Progress: "))
    
    # Summary
    total_duration = datetime.now() - start_time
    successes = sum(1 for r in results if r[2])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_duration}")
    print(f"Passed: {successes}/{total}")
    print()
    
    # Results table
    print(f"{'Variant':<25} {'Task':<12} {'Status':<8} {'Duration':<10}")
    print("-" * 60)
    for variant, task, success, message, duration in results:
        variant_short = variant.split("/")[-1]
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{variant_short:<25} {task:<12} {status:<8} {duration.total_seconds():.1f}s")
    
    print("=" * 70)
    
    # Exit with error if any failed
    sys.exit(0 if successes == total else 1)


if __name__ == "__main__":
    main()
