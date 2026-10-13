"""C5B1-R2 pure, fixture-only geometry tests: no native backend or artifacts."""
import unittest
from src.p2b3_c5b1_r2_attribution import metrics, threshold_root, THRESHOLDS_M

class TestC5B1R2Attribution(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"p":(0.0,0.0,0.0), "a":(0.0,0.0,1.0),
             "t":(0.0,0.0,0.0), "n":(0.0,0.0,1.0)}
            for _ in range(181)
        ]

    def test_tangent_motion_not_standoff(self):
        m=metrics(self.rows,"ty",1,.006)
        self.assertAlmostEqual(m["max_position_error_m"]["value"],.006)
        self.assertAlmostEqual(m["max_abs_plane_axial_standoff_delta_m"]["value"],0.0)
        self.assertAlmostEqual(m["max_plane_lateral_offset_m"]["value"],.006)
        self.assertEqual(m["max_material_normal_angle_deg"]["value"],0.0)
        self.assertEqual(m["quality_failure"],"NOT_ASSESSED")

    def test_normal_motion_changes_standoff(self):
        m=metrics(self.rows,"tz",1,.006)
        self.assertAlmostEqual(m["max_abs_plane_axial_standoff_delta_m"]["value"],.006)
        self.assertAlmostEqual(m["max_plane_lateral_offset_m"]["value"],0.0)

    def test_threshold_changes_are_not_physical_failure(self):
        for gate in THRESHOLDS_M:
            root=threshold_root(self.rows,"tx",1,gate)
            self.assertEqual(root["status"],"ANALYTIC_ENGINEERING_THRESHOLD_CROSSING")
            self.assertAlmostEqual(sum(root["bracket"])/2,gate,places=9)
            self.assertTrue(root["not_a_physical_failure"])

    def test_backward_facing_ray_fails_closed(self):
        rows=[{"p":(0,0,0),"a":(0,0,-1),"t":(0,0,0),"n":(0,0,1)}
              for _ in range(181)]
        m=metrics(rows,"ty",1,.006)
        self.assertEqual(m["ray_hit_proxy_status"],"INDETERMINATE_RAY_INTERSECTION")
        self.assertIsNone(m["max_plane_lateral_offset_m"])

    def test_rotation_base_origin(self):
        rows=[{"p":(1,0,0),"a":(0,0,1),"t":(1,0,0),"n":(0,0,1)}
              for _ in range(181)]
        m=metrics(rows,"rz",1,.01)
        self.assertGreater(m["max_position_error_m"]["value"],0.009)
        self.assertLess(m["max_position_error_m"]["value"],0.011)

if __name__=="__main__":
    unittest.main()
