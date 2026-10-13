"""Finite C1 extruded-wall intersection tests; no FCL/ROS runtime."""
import math
import unittest

from src.p2b3_c5b1_r3_mesh import (
    closest_segment, intersect_open_wall, projected_normal_diagnostic,
    evaluate_direction, wall,
)


def section(y=0., dx=0.):
    rows = []
    for i in range(181):
        x = i * .01
        rows.append(dict(t=(x,y,0.), n=(0.,0.,1.),
                         p=(x+dx,y,0.), a=(0.,0.,1.)))
    return rows


class R3MeshTests(unittest.TestCase):
    def test_forward_surface_is_real_finite_segment_not_infinite_plane(self):
        rows = section(dx=.002)
        r = intersect_open_wall(rows, "tz", 1, .006, 50)
        self.assertEqual(r["status"], "HIT")
        self.assertAlmostEqual(r["first_hit"]["distance_m"], .266, places=9)
        self.assertAlmostEqual(r["first_hit_axial_standoff_delta_m"], .006, places=9)
        self.assertEqual(r["first_hit"]["segment"], 50)

    def test_missing_surface_hit_not_interpreted_as_pass(self):
        rows = section(dx=.002)
        rows[50]["p"] = (20.,0.,0.)
        r=intersect_open_wall(rows,"tz",1,.006,50)
        self.assertEqual(r["status"],"NO_FORWARD_SURFACE_HIT")
        self.assertIsNone(r["first_hit"])

    def test_outside_finite_y_span_is_no_hit(self):
        rows=section(dx=.002)
        rows[50]["p"]=(.502,1.,0.)
        self.assertEqual(intersect_open_wall(rows,"tz",1,.006,50)["status"],
                         "NO_FORWARD_SURFACE_HIT")

    def test_ray_points_away_from_wall(self):
        rows=section(dx=.002)
        rows[50]["a"]=(0.,0.,-1.)
        self.assertEqual(intersect_open_wall(rows,"tz",1,.006,50)["status"],
                         "NO_FORWARD_SURFACE_HIT")

    def test_nearest_association_reacts_to_translation(self):
        rows=section(dx=.002)
        x=projected_normal_diagnostic(rows,"tz",1,.006)
        self.assertAlmostEqual(x["critical"]["projected_angle_deg"],0.,places=8)
        self.assertAlmostEqual(x["critical"]["same_material_angle_deg"],0.,places=8)

    def test_finite_surface_fail_closed(self):
        rows=section(dx=.002)
        rows[50]["p"]=(20.,0.,0.)
        result=evaluate_direction(rows,"tz",1,.006)
        self.assertEqual(result["intersection_status"],"INDETERMINATE")
        self.assertIn(50,result["invalid_waypoints"])
        self.assertEqual(result["coating_quality"],"NOT_ASSESSED")

    def test_rotation_raycast_has_coordinate_covariance(self):
        rows=section(dx=.002)
        a=intersect_open_wall(rows,"rz",1,math.radians(.5),50)
        b=intersect_open_wall(rows,"rz",-1,math.radians(.5),50)
        self.assertEqual(a["status"],"HIT")
        self.assertEqual(b["status"],"HIT")
        self.assertTrue(math.isfinite(a["first_hit"]["distance_m"]))
        self.assertTrue(math.isfinite(b["first_hit"]["distance_m"]))

    def test_wall_requires_exact_archived_cardinality(self):
        with self.assertRaisesRegex(ValueError,"181"):
            wall(section()[:180])


if __name__=="__main__":
    unittest.main()
