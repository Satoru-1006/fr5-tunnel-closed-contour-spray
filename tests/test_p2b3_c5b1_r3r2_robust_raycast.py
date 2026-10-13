"""FR5 C5B1 R3-R2 bounded surface raycast adversarial tests.

These are synthetic geometry fixtures, not a physical-spray or native FCL test.
"""
import math
import unittest

from src.p2b3_c5b1_r3r2_robust_raycast import (
    _facet_intersections, _group_adjacent_hits, evaluate_direction, intersect_waypoint,
    wall_points,
)


def rows(kink=False):
    data=[]
    for i in range(181):
        x=i*0.01
        z=max(0,i-90)*0.001 if kink else 0.
        data.append({"p":(x,0.,z),"t":(x,0.,z),"n":(0.,0.,1.),"a":(0.,0.,1.)})
    return data


class RobustRaycastTest(unittest.TestCase):
    def test_baseline_flat_finite_standoff(self):
        r=rows(); h=intersect_waypoint(r,wall_points(r),50,'tz',1,.006)
        self.assertEqual(h['position_status'],'FIRST_HIT_POSITION_DETERMINATE')
        self.assertEqual(h['normal_status'],'UNIQUE_FACET_NORMAL')
        self.assertAlmostEqual(h['first_hit']['axial_standoff_delta_m'],.006,places=10)

    def test_shared_cross_section_edge_position_known_normal_nonunique(self):
        r=rows(kink=True)
        h=intersect_waypoint(r,wall_points(r),90,'tz',1,0.)
        self.assertEqual(h['position_status'],'FIRST_HIT_POSITION_DETERMINATE')
        self.assertEqual(h['normal_status'],'AMBIGUOUS_SHARED_EDGE_NORMAL')
        self.assertEqual(h['first_hit']['participating_segment_ids'],[89,90])
        self.assertGreater(h['first_hit']['facet_normal_separation_deg'],5.)
        self.assertAlmostEqual(h['first_hit']['normal_interpolated_angle_interval_deg'][1],0.,places=8)

    def test_complete_positions_do_not_imply_unique_normals(self):
        r=rows(kink=True)
        s=evaluate_direction(r,'tz',1,0.)
        self.assertEqual(s['position_certainty'],'COMPLETE')
        self.assertEqual(s['normal_certainty'],'PARTIAL_OR_AMBIGUOUS')
        self.assertIn(90,s['shared_edge_multiple_normal_waypoints'])

    def test_no_forward_hit_not_success(self):
        r=rows(); r[70]['p']=(99.,0.,0.)
        s=evaluate_direction(r,'tz',1,0.)
        self.assertEqual(s['position_certainty'],'PARTIAL_OR_INDETERMINATE')
        self.assertIn(70,s['no_forward_hit_waypoints'])
        self.assertEqual(s['normal_certainty'],'PARTIAL_OR_AMBIGUOUS')

    def test_distant_coplanar_support_plane_not_false_block(self):
        # The final finite vertical facet lies BEHIND the ray origin.
        section=[(0.,0.,.26),(1.,0.,.26),(.5,0.,-2.),(.5,0.,-1.)]
        hit,over=_facet_intersections(section,[(0.,0.,1.)]*len(section),(.5,0.,0.),(0.,0.,1.))
        self.assertTrue(hit)
        self.assertFalse(over)
        self.assertAlmostEqual(hit[0]['distance_m'],.26,places=9)

    def test_coplanar_interval_after_first_hit_cannot_override(self):
        section=[(0.,0.,.26),(1.,0.,.26),(.5,0.,.27),(.5,0.,.47)]
        hit,over=_facet_intersections(section,[(0.,0.,1.)]*len(section),(.5,0.,0.),(0.,0.,1.))
        self.assertTrue(hit)
        self.assertTrue(over)
        self.assertGreater(over[0]['t_interval_m'][0],hit[0]['distance_m'])

    def test_coplanar_interval_before_first_is_uncertain(self):
        section=[(0.,0.,.26),(1.,0.,.26),(.5,0.,.20),(.5,0.,.30)]
        hit,over=_facet_intersections(section,[(0.,0.,1.)]*len(section),(.5,0.,0.),(0.,0.,1.))
        self.assertTrue(hit)
        self.assertTrue(over)
        self.assertLess(over[0]['t_interval_m'][0],hit[0]['distance_m'])

    def test_nonadjacent_close_hits_not_collapsed(self):
        a={"segment":5,"distance_m":.26,"at_cross_section_edge":True}
        b={"segment":20,"distance_m":.26+1e-11,"at_cross_section_edge":True}
        self.assertEqual(len(_group_adjacent_hits([a,b])),2)

    def test_adjacent_shared_edge_hits_collapsed_geometrically(self):
        a={"segment":5,"distance_m":.26,"at_cross_section_edge":True}
        b={"segment":6,"distance_m":.26+1e-12,"at_cross_section_edge":True}
        self.assertEqual(len(_group_adjacent_hits([a,b])),1)

    def test_adjacent_interior_crossings_are_not_merged(self):
        a={"segment":5,"distance_m":.26,"at_cross_section_edge":False}
        b={"segment":6,"distance_m":.26+1e-12,"at_cross_section_edge":False}
        self.assertEqual(len(_group_adjacent_hits([a,b])),2)

    def test_rotation_inverse_coordinates_consistency(self):
        r=rows(); sec=wall_points(r)
        x=intersect_waypoint(r,sec,50,'rz',1,math.radians(.2))
        y=intersect_waypoint(r,sec,50,'rz',-1,math.radians(.2))
        self.assertEqual(x['position_status'],'FIRST_HIT_POSITION_DETERMINATE')
        self.assertEqual(y['position_status'],'FIRST_HIT_POSITION_DETERMINATE')

    def test_near_parallel_outside_plane_not_spurious_hit(self):
        sec=[(0.,0.,.26),(1.,0.,.26)]
        hit,over=_facet_intersections(sec,[(0.,0.,1.)]*2,(.5,0.,0.),(1.,0.,1e-14))
        self.assertEqual(hit,[])
        self.assertEqual(over,[])

    def test_longitudinal_limit_is_finite(self):
        r=rows()
        r[50]['p']=(.50,.551,0.)
        h=intersect_waypoint(r,wall_points(r),50,'tz',1,0.)
        self.assertEqual(h['position_status'],'NO_FORWARD_FINITE_SURFACE_HIT')
        r[50]['p']=(.50,.55,0.)
        h=intersect_waypoint(r,wall_points(r),50,'tz',1,0.)
        self.assertEqual(h['position_status'],'FIRST_HIT_POSITION_DETERMINATE')

    def test_nonadjacent_multiple_forward_hits_retained(self):
        sec=[(0.,0.,.26),(1.,0.,.26),(2.,0.,.45),(1.,0.,.60),(0.,0.,.60)]
        hit,over=_facet_intersections(sec,[(0.,0.,1.)]*len(sec),(.5,0.,0.),(0.,0.,1.))
        gg=_group_adjacent_hits(hit)
        self.assertEqual(len(gg),2)
        self.assertLess(min(v['distance_m'] for v in gg[0]),min(v['distance_m'] for v in gg[1]))

    def test_no_physical_certification_flag_in_synthetic(self):
        s=evaluate_direction(rows(),'tz',1,.006)
        self.assertEqual(s['physical_coating_quality'],'NOT_ASSESSED')
        self.assertEqual(s['continuous_collision'],'NOT_ASSESSED')


if __name__=='__main__':unittest.main()
