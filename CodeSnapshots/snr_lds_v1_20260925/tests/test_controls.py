import importlib.util
from pathlib import Path
import unittest
import tempfile
import numpy as np
from scipy import stats

from balds.workflows import controls as m

class Tests(unittest.TestCase):
    def test_head_and_monotone(self):
        s=np.array([[-2.,1.],[1.,1.],[1.,-1.],[0.,2.]])
        np.testing.assert_array_equal(m.head(s,2)[:,0],[False,True,True,False])
        np.testing.assert_array_equal(m.head(s,2),m.head(np.sign(s)*s*s,2))
    def test_ties(self):
        y=np.array([[0.],[1.],[2.]])
        self.assertEqual(m.pair_loss(y,y)[0][0],0)
        self.assertEqual(m.pair_loss(-y,y)[0][0],1)
        self.assertEqual(m.pair_loss(np.zeros_like(y),y)[0][0],.5)
        self.assertTrue(np.isnan(m.pair_loss(y,np.zeros_like(y))[0][0]))
    def test_selection_ties(self):
        l=np.array([[1.,1.],[1.,0.]])
        u=np.array([[2.,3.],[4.,9.]])
        np.testing.assert_equal(m.choose(l,u)[0],[3.,3.])
    def test_count_decomposition(self):
        s=np.random.default_rng(4).normal(size=(20,3))
        d=np.random.default_rng(5).integers(0,2,(8,20))
        np.testing.assert_allclose(d@s,d@(s-s.mean(axis=0))+d@np.broadcast_to(s.mean(axis=0),s.shape))
    def test_ci_paired(self):
        r=m.ci(np.zeros(50))
        self.assertEqual(r,dict(mean=0.,ci_low=0.,ci_high=0.))
    def test_literal_tabs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.tsv";p.write_text("query_id\\tk\n0\\t300\n")
            self.assertEqual(m.read_table(p),[dict(query_id="0",k="300")])
    def test_native_metric(self):
        d=np.random.default_rng(3).integers(0,2,(8,12))
        s=np.random.default_rng(4).normal(size=(12,3));y=np.random.default_rng(5).normal(size=(8,3))
        p,l,*_=m.metrics(s,d,y)
        np.testing.assert_allclose(l,[stats.spearmanr(y[:,q],p[:,q]).statistic for q in range(3)])

if __name__=="__main__":
    unittest.main()
