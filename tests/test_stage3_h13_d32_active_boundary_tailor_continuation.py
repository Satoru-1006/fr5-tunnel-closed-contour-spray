import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import stage3_h13_post_d31_d32_active_boundary_tailor_continuation as d32


def scalar_map(first: float, second: float = 0.0):
    values = {name: torch.zeros(1, dtype=torch.float64) for name in d32.d24.d16.ALL_NAMES}
    values[d32.d24.d16.ALL_NAMES[0]] = torch.tensor([first], dtype=torch.float64)
    values[d32.d24.d16.ALL_NAMES[1]] = torch.tensor([second], dtype=torch.float64)
    return values


def test_tangent_projection_removes_normal_component():
    values = scalar_map(3.0, 4.0)
    normal = scalar_map(1.0, 0.0)
    tangent, coefficient = d32.project_tangent(values, normal)
    assert coefficient == 3.0
    assert abs(d32.dot(tangent, normal)) < 1.0e-12
    assert d32.norm(tangent.values()) == 4.0


def test_conflict_projection_preserves_already_inward_gradient():
    normal = scalar_map(1.0, 0.0)
    inward = scalar_map(2.0, 1.0)
    result, coefficient = d32.project_only_conflicting(inward, normal)
    assert coefficient == 0.0
    assert all(torch.equal(result[name], inward[name]) for name in result)
    conflicting = scalar_map(-2.0, 1.0)
    repaired, coefficient = d32.project_only_conflicting(conflicting, normal)
    assert coefficient == -2.0
    assert abs(d32.dot(repaired, normal)) < 1.0e-12


def test_candidate_rank_prefers_inward_boundary_class():
    base = {
        "candidate_H1": 1.0,
        "H32_margin": 1.0e-6,
        "effective_update_norm": 1.0,
        "spec_id": "a",
    }
    inward = {**base, "delta_H32": -1.0e-6}
    flat = {**base, "delta_H32": 0.0, "candidate_H1": 0.5}
    outward = {**base, "delta_H32": 1.0e-6, "candidate_H1": 0.1}
    assert d32.candidate_rank(inward) < d32.candidate_rank(flat) < d32.candidate_rank(outward)


def test_spec_identity_distinguishes_h1_bias_and_cagrad_controls():
    base = {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "alpha": 0.01, "lambda": 0.0}
    assert d32.spec_id(base) != d32.spec_id({**base, "p4_beta": 1.4})
    cagrad = {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "CAGRAD", "alpha": 0.01, "lambda": 0.0}
    assert d32.spec_id(cagrad) != d32.spec_id({**cagrad, "cagrad_c": 0.8})


def test_adamw_predictor_matches_torch_with_inherited_moments():
    names = d32.d24.d16.ALL_NAMES
    before_parameters = {name: torch.tensor([0.2 + index * 0.01], dtype=torch.float64) for index, name in enumerate(names)}
    gradients = {name: torch.tensor([0.03 - index * 0.001], dtype=torch.float64) for index, name in enumerate(names)}
    before_optimizer = {
        name: {
            "step": torch.tensor(7.0),
            "exp_avg": torch.tensor([0.01 + index * 0.0001], dtype=torch.float64),
            "exp_avg_sq": torch.tensor([0.02 + index * 0.0002], dtype=torch.float64),
        }
        for index, name in enumerate(names)
    }
    lr = 2.5e-6
    predicted = d32.predicted_adamw_delta(before_parameters, before_optimizer, gradients, lr)
    parameters = [torch.nn.Parameter(before_parameters[name].clone()) for name in names]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=lr,
        betas=d32.consistency.BETAS,
        eps=d32.consistency.EPS,
        weight_decay=d32.consistency.WEIGHT_DECAY,
    )
    for name, parameter in zip(names, parameters):
        optimizer.state[parameter] = copy.deepcopy(before_optimizer[name])
        parameter.grad = gradients[name].clone()
    optimizer.step()
    for name, parameter in zip(names, parameters):
        realized = parameter.detach() - before_parameters[name]
        assert torch.allclose(predicted[name], realized, rtol=1.0e-9, atol=1.0e-12)
