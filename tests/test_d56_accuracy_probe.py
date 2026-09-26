from __future__ import annotations

import numpy as np

from tools.d56_accuracy_probe import project_task_path, quat_z_axis


def test_quaternion_z_axis_matches_authoritative_target_normal() -> None:
    quaternion = np.asarray((0.951494740377893, 5.826032523476705e-17, 0.3076650110643166, -1.8838426369909706e-17))
    expected = np.asarray((0.585483279652007, 7.169865443186115e-17, -0.8106844819335879))

    np.testing.assert_allclose(quat_z_axis(quaternion), expected, atol=1.0e-12)


def test_task_projection_is_geometric_and_interpolates_normals() -> None:
    target_position = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    target_normal = np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)))
    actual_position = np.asarray(((0.5, 0.02, 0.0),))

    projected_position, projected_normal, distance = project_task_path(actual_position, target_position, target_normal)

    np.testing.assert_allclose(projected_position, ((0.5, 0.0, 0.0),))
    np.testing.assert_allclose(projected_normal, ((0.0, 0.0, 1.0),))
    np.testing.assert_allclose(distance, (0.02,))
