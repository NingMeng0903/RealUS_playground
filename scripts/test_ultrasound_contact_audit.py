"""Engineering checks, not validation against physical contact ground truth."""
import unittest
import numpy as np
from ultrasound_contact_audit import bands, nearest
from ultrasound_contact_report import spatial_width, episodes


class ContactAuditChecks(unittest.TestCase):
    def test_deep_shadow_is_not_nearfield_dropout(self):
        im=np.full((726,664),140,np.uint8)
        im[330:,:250]=0
        occ=bands(im)[None]
        width,_,_=spatial_width(occ)
        self.assertEqual(float(width[0]),0)

    def test_edge_dropout_has_location_and_width(self):
        im=np.full((726,664),140,np.uint8)
        im[25:400,:180]=0
        width,side,_=spatial_width(bands(im)[None])
        self.assertGreater(float(width[0]),.125)
        self.assertEqual(side[0],'left')

    def test_both_edges_not_mislabelled_as_one_side(self):
        occ=np.zeros((1,3,32)); occ[:,:,:5]=1;occ[:,:,-5:]=1
        _,side,_=spatial_width(occ)
        self.assertEqual(side[0],'both')

    def test_short_dark_transient_rejected(self):
        t=np.arange(30)/30; m=np.zeros(30,bool);m[10:13]=True
        keep,rr=episodes(t,m)
        self.assertFalse(keep.any());self.assertEqual(rr,[])

    def test_persistence_bridges_only_short_gaps(self):
        t=np.arange(40)/30;m=np.zeros(40,bool);m[5:13]=True;m[15:23]=True;m[30:38]=True
        _,rr=episodes(t,m)
        self.assertEqual(rr,[(5,23),(30,38)])

    def test_nearest_tie_prefers_earlier(self):
        t=np.array([100,110,120],np.int64)
        np.testing.assert_array_equal(nearest(t,np.array([105,116])),[0,2])


if __name__=='__main__':
    unittest.main()
