#!/bin/bash
# Setup script for W&B online syncing
# Usage: bash scripts/setup_wandb.sh

echo "=== W&B Setup ==="
echo ""
echo "Step 1: Login to W&B"
echo "  Go to https://wandb.ai/authorize to get your API key"
echo ""

read -p "Paste your W&B API key: " WANDB_KEY

if [ -z "$WANDB_KEY" ]; then
    echo "No key provided. Exiting."
    exit 1
fi

# Login
wandb login "$WANDB_KEY"

if [ $? -eq 0 ]; then
    echo ""
    echo "Step 2: Syncing offline run..."
    wandb sync wandb/offline-run-20260416_200554-d99ue25v
    echo ""
    echo "Done! Check your W&B dashboard at https://wandb.ai"
else
    echo "Login failed. Try setting the env var instead:"
    echo "  export WANDB_API_KEY=your_key_here"
    echo "  wandb sync wandb/offline-run-20260416_200554-d99ue25v"
fi
