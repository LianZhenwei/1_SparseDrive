from pathlib import Path

cfg = Path("projects/configs/sparsedrive_small_stage2_mini.py")
text = cfg.read_text()

text = text.replace(
    "version = 'mini'\nversion = 'trainval'",
    "version = 'mini'\n# version = 'trainval'"
)

text = text.replace(
    "total_batch_size = 48\nnum_gpus = 8\nbatch_size = total_batch_size // num_gpus",
    "total_batch_size = 1\nnum_gpus = 1\nbatch_size = 1"
)

text = text.replace(
    'version="v1.0-trainval"',
    'version="v1.0-mini"'
)

cfg.write_text(text)
print(f"patched {cfg}")