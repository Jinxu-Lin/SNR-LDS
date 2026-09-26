import numpy as np

from balds.evaluation.lds import support_head_scores


def test_support_head_preserves_native_values_and_breaks_boundary_ties_by_index():
    scores = np.array([[3.0, 1.0], [3.0, 4.0], [2.0, 4.0], [-1.0, 0.0]],
                      dtype=np.float32)
    head = support_head_scores(scores, 0.5)
    # q0: rows 0 and 1 are tied and both selected. q1: rows 1 and 2.
    want = np.array([[3.0, 0.0], [3.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
                    dtype=np.float32)
    assert np.array_equal(head, want)
    assert np.array_equal(support_head_scores(scores, 1.0), scores)


def test_support_head_boundary_tie_uses_first_training_row():
    scores = np.ones((4, 1), dtype=np.float32)
    head = support_head_scores(scores, 0.25)
    assert head[:, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
