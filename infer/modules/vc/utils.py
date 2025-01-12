import os

from fairseq import checkpoint_utils
from pathlib import Path


def get_index_path_from_model(sid):
    s = Path(sid)
    s = s.stem.rsplit("_e")[0]
    print(s)
    return next(
        (
            f.replace('/','\\')
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]+[
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("outside_index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if s in f
        ),
        "",
    )


def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
