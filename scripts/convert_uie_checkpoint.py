"""Extract tensors from the upstream UIE object checkpoint with inert objects."""
import collections
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import BertConfig, BertModel, ErnieConfig, ErnieModel

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / ".cache/models/uie-base-zh"


class Container:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
            ("builtins", "set"): set,
            ("__builtin__", "set"): set,
        }
        if (module, name) in allowed:
            return allowed[module, name]
        if module.startswith(("torch.nn.modules.", "transformers.models.", "transformers.activations")) or (module, name) in [("model", "UIE"), ("torch.nn.functional", "gelu")]:
            return Container
        raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}")


def tensors(obj, prefix=""):
    result = {}
    for group in ["_parameters", "_buffers"]:
        for name, value in getattr(obj, group, {}).items():
            if isinstance(value, torch.Tensor):
                result[prefix + name] = value.detach().cpu().contiguous()
    for name, child in getattr(obj, "_modules", {}).items():
        if child is not None:
            result.update(tensors(child, prefix + name + "."))
    return result


def main():
    path = FOLDER / "pytorch_model.bin"
    expected = "1a3e9c51156faafe74bbc080d31d1c430708231a8c86c458924da13ccba488b7"
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual == expected
    inert_pickle = types.SimpleNamespace(Unpickler=RestrictedUnpickler, __name__="inert_checkpoint")
    original = torch.load(path, map_location="cpu", pickle_module=inert_pickle, weights_only=False)
    state = tensors(original)
    print("Extracted", len(state), "tensors", flush=True)
    encoder = original._modules["encoder"]
    config = encoder.config.__dict__.copy()
    config = {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    model_type = config.get("model_type", "bert")
    cls, constructor = (ErnieConfig, ErnieModel) if "ernie" in model_type or any("task_type_embeddings" in k for k in state) else (BertConfig, BertModel)
    settings = cls.from_dict(config)
    backbone = constructor(settings)
    weights = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = backbone.load_state_dict(weights, strict=False)
    # Non-persistent position/token-type buffers were saved by older transformers.
    unexpected = [k for k in unexpected if k not in ["embeddings.position_ids", "embeddings.token_type_ids"]]
    assert not missing and not unexpected, (missing, unexpected)
    converted = FOLDER / "converted"
    backbone.save_pretrained(converted, safe_serialization=True)
    boundary = {k: state[k] for k in ["linear_start.weight", "linear_start.bias", "linear_end.weight", "linear_end.bias"]}
    save_file(boundary, str(converted / "boundary.safetensors"))
    (converted / "conversion.json").write_text(json.dumps({"source": "Pky/uie-base-zh", "sha256": actual,
                "tensors": len(state), "encoder": settings.to_dict(), "missing": missing, "unexpected": unexpected,
                "loader": "restricted unpickler with inert module containers; original checkpoint code is never invoked"}, indent=2), encoding="utf-8")
    print("Converted pretrained UIE encoder and boundary heads", flush=True)


if __name__ == "__main__":
    main()
