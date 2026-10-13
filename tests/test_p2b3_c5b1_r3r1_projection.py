"""Pure-function R3-R1 fixture tests, no native FR5 backend needed."""
import math
import unittest
from src.p2b3_c5b1_r3r1_projection import (
    all_candidates, angle_deg, build_segments, critical_probe,
    crossing_bisection, pair_squared_difference, worst_angle,
)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        # Two perpendicular open edges with two *interior* nearest feet.
        self.pts=((0.,0.,0.),(1.,0.,0.),(1.,0.,1.))
        self.ns=((0.,0.,1.),(0.,0.,1.),(1.,0.,0.))
        self.axis=(0.,0.,1.)
        self.p=(.875,0.,.125)

    def test_candidate_first_winner_at_exact_tie(self):
        out=critical_probe(self.p,self.axis,build_segments(self.pts,self.ns),0.,5)
        self.assertEqual(out['winner']['segment'],0)
        self.assertEqual(out['runner_up']['segment'],1)
        self.assertAlmostEqual(out['runner_up_minus_winner_distance_m'],0.,places=13)
        self.assertTrue(out['near_tie_diagnostic'])

    def test_normal_difference_can_jump_when_distance_ties(self):
        seg=build_segments(self.pts,self.ns)
        a=critical_probe(self.p,self.axis,seg,-1e-5,5)
        b=critical_probe(self.p,self.axis,seg,1e-5,5)
        self.assertNotEqual(a['winner']['segment'],b['winner']['segment'])
        self.assertGreater(abs(a['winner']['angle_deg']-b['winner']['angle_deg']),3.)

    def test_pair_bisection_locates_zero(self):
        seg=build_segments(self.pts,self.ns)
        x=crossing_bisection(self.p,self.axis,seg,-1e-3,1e-3,0,1)
        self.assertEqual(x['status'],'PAIR_EQUAL_DISTANCE_CROSSING')
        self.assertLess(abs(x['signed_tz_root_m']),1e-9)
        self.assertNotEqual(x['left_winner'],x['right_winner'])

    def test_no_pair_root_fails_closed(self):
        seg=build_segments(self.pts,self.ns)
        x=crossing_bisection(self.p,self.axis,seg,0.01,0.02,0,1)
        self.assertEqual(x['status'],'NO_PAIR_SIGN_CROSSING_IN_INTERVAL')

    def test_straight_subdivision_preserves_nearest_physical_distance(self):
        for split in (2,4,8):
            s=build_segments(self.pts,self.ns,subdivide=split)
            out=critical_probe((.91,0,.09),self.axis,s,0.,0)
            original=critical_probe((.91,0,.09),self.axis,
                                    build_segments(self.pts,self.ns),0.,0)
            self.assertAlmostEqual(out['winner']['distance_m'],original['winner']['distance_m'],places=12)
            self.assertEqual(out['winner']['segment'],original['winner']['segment'])

    def test_invalid_configuration_fails(self):
        with self.assertRaises(ValueError):build_segments(self.pts,self.ns,subdivide=0)

    def test_distinct_parent_runner_up(self):
        s=build_segments(self.pts,self.ns,subdivide=8)
        out=critical_probe((.5,0,.05),self.axis,s,0.,0)
        self.assertNotEqual(out['winner']['segment'],out['runner_up']['segment'])

    def test_negative_distance_gap_disallowed(self):
        s=build_segments(self.pts,self.ns)
        out=critical_probe(self.p,self.axis,s,1e-3,0)
        self.assertGreaterEqual(out['runner_up_minus_winner_distance_m'],-1e-12)


if __name__=='__main__':unittest.main()
