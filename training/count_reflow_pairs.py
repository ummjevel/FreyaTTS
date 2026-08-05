"""Count reflow pairs across shards, so a chain can verify generation finished.

Kept as a file rather than an inline heredoc: the shell scripts that call this
already use heredocs, and nesting them silently truncates the outer one.
"""
import glob
import sys

import torch

d = sys.argv[1]
print(sum(len(torch.load(f, weights_only=False)) for f in glob.glob(f"{d}/*.pt")))
