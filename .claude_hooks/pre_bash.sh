#!/bin/bash
CMD="$1"
# GPU memory guard — warn if running a training command with low memory
if echo "$CMD" | grep -q "train.py"; then
  python -c "
import torch
if torch.cuda.is_available():
    free = torch.cuda.mem_get_info()[0] / 1e9
    total = torch.cuda.mem_get_info()[1] / 1e9
    print(f'GPU: {free:.1f}GB free / {total:.1f}GB total')
    if free < 4.0:
        print('WARNING: Low GPU memory — consider reducing minibatch_size in config')
"
fi
