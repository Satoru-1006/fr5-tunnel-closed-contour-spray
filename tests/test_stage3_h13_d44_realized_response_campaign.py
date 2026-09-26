import numpy as np

from tools.stage3_h13_d44_realized_response_campaign import (
    H32_LIMIT,
    SurrogateEnsemble,
    model_features,
    relative_violation,
)


def _horizons(h1: float, h32: float, retained: float = 1.0) -> dict[str, float]:
    values = {str(h): retained for h in range(1, 33)}
    values["1"] = h1
    values["32"] = h32
    return values


def test_relative_violation_separates_h32_and_retention() -> None:
    parent = _horizons(1.0, H32_LIMIT - 1.0e-6)
    safe = _horizons(0.9, H32_LIMIT - 2.0e-6)
    assert relative_violation(safe, parent, parent) == 0.0
    h32_unsafe = _horizons(0.9, H32_LIMIT + 1.0e-4)
    assert relative_violation(h32_unsafe, parent, parent) > 0.0
    retained_unsafe = _horizons(0.9, H32_LIMIT - 1.0e-6, retained=1.1)
    assert relative_violation(retained_unsafe, parent, parent) > 0.0


def test_surrogate_competition_returns_finite_mult_output_prediction() -> None:
    rows = []
    for index in range(6):
        z = np.zeros(3, dtype=float)
        z[index % 3] = 1.0
        x = model_features(z, 0.00390625, 3)
        rows.append({"x": x.tolist(), "delta": [-(index + 1) * 1.0e-7, 0.0, (index % 2) * 1.0e-5]})
    ensemble = SurrogateEnsemble(output_dim=3, input_dim=4)
    ensemble.fit(rows, trust_radius=1.0)
    mean, uncertainty, errors = ensemble.predict(model_features([0.2, 0.3, 0.1], 0.00390625, 3))
    assert mean.shape == (3,)
    assert uncertainty.shape == (3,)
    assert np.isfinite(mean).all()
    assert np.isfinite(uncertainty).all()
    assert set(errors) >= {"local_linear", "rbf"}

