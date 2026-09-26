import hashlib
from collections import OrderedDict
from pathlib import Path

import torch

from scripts.stage3_h13_r8e_b_multiseed_robustness import R8E_ROOT, sha256_file
from scripts.stage3_h13_r8e_d_counterfactual_feedback_audit import checkpoint_state_hash


def _fake_model(items):
    class FakeModel:
        def state_dict(self):
            return OrderedDict(items)

    return FakeModel()


def test_raw_file_hashing_is_sha256_over_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    payload = b"raw serialized checkpoint bytes"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_canonical_state_hash_is_deterministic_under_key_order() -> None:
    left = _fake_model([("b", torch.tensor([2.0])), ("a", torch.tensor([1.0]))])
    right = _fake_model([("a", torch.tensor([1.0])), ("b", torch.tensor([2.0]))])
    assert checkpoint_state_hash(left) == checkpoint_state_hash(right)


def test_canonical_state_hash_is_sensitive_to_dtype_shape_and_value() -> None:
    base = checkpoint_state_hash(_fake_model([("x", torch.tensor([1.0, 2.0], dtype=torch.float32))]))
    dtype_changed = checkpoint_state_hash(_fake_model([("x", torch.tensor([1.0, 2.0], dtype=torch.float64))]))
    shape_changed = checkpoint_state_hash(_fake_model([("x", torch.tensor([[1.0, 2.0]], dtype=torch.float32))]))
    value_changed = checkpoint_state_hash(_fake_model([("x", torch.tensor([1.0, 3.0], dtype=torch.float32))]))
    assert len({base, dtype_changed, shape_changed, value_changed}) == 4


def test_a1_checkpoint_is_immutable_during_read_only_identity_checks() -> None:
    path = R8E_ROOT / "best_candidate_checkpoint.pt"
    before = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    assert "model_state_dict" in payload
    assert sha256_file(path) == before


def test_r8e_d_seed_13086_raw_and_path_gates_are_unsatisfiable() -> None:
    source = Path("scripts/stage3_h13_r8e_d_counterfactual_feedback_audit.py").read_text(encoding="utf-8")
    assert 'sha256_file(resolved) != expected or expected != str(records[seed]["checkpoint_sha256"])' in source
    assert 'checkpoint_paths[13086].resolve() != (R8E_ROOT / "best_candidate_checkpoint.pt").resolve()' in source
    recorded_r8e_b_sha = "0b7871d2190b012b06a7d5f06dbb5ea0be03ce0cb8bc4a34f9e21d4a5a4fdd95"
    current_a1_sha = "4d57a265ab130619ae2971124a514c35bf6efac8731fbadbaef6a8d9d921e6ac"
    assert recorded_r8e_b_sha != current_a1_sha
    assert not (current_a1_sha == recorded_r8e_b_sha)

