#!/bin/bash

# Overnight evaluation script
# Runs each evaluation sequentially, continues on error

echo "Starting overnight evaluation run at $(date)"
echo "=========================================="

run_eval() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: python run_eval.py $@"
    echo "------------------------------------------"
    python run_eval.py "$@" || echo "[ERROR] Command failed, continuing to next..."
    echo ""
}


run_eval --object microwave/7236_ivw --task tool_use --episodes 5
run_eval --object microwave/7236_vlm1shot --task tool_use --episodes 5


run_eval --object laptop/9912_evogen --task tool_use --episodes 5
run_eval --object laptop/9912_humangensim2 --task tool_use --episodes 5
run_eval --object laptop/9912_ivw --task tool_use --episodes 5
run_eval --object laptop/9912_vlm1shot --task tool_use --episodes 5

# run_eval --object cabinet/7236_evogen --task tool_use --episodes 5
# run_eval --object cabinet/7236_humangensim2 --task tool_use --episodes 5
# run_eval --object cabinet/7236_ivw --task tool_use --episodes 5
# run_eval --object cabinet/7236_vlm1shot --task tool_use --episodes 5

echo "=========================================="
echo "Overnight evaluation completed at $(date)"
