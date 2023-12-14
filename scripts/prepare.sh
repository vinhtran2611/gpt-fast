python3 scripts/download.py --repo_id $1 && python3 scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$1 && python3 quantize.py --checkpoint_path checkpoints/$1/model.pth --mode int8
