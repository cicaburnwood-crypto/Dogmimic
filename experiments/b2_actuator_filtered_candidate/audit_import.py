import json
from pathlib import Path
import train_guarded,train_robust,robust_env
assert Path(robust_env.__file__).resolve().parent==Path(__file__).resolve().parent
assert train_guarded.base.WorldModel is train_guarded.FilteredWM
assert train_robust.XI_DIM==43
print(json.dumps(dict(passed=True,environment_file=robust_env.__file__,replay_context_dim=train_robust.XI_DIM,world_model=train_guarded.base.WorldModel.__name__)))
