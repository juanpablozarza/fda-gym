from fdagym.train.advantage import advantages


def test_equal_rewards_are_all_zero():
    assert advantages([1.0, 1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0, 0.0]
    assert advantages([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_mixed_group_centers_on_the_mean():
    values = advantages([1.0, 0.1, 0.1, 0.1])
    assert values[0] > 0
    assert all(v < 0 for v in values[1:])
    assert abs(sum(values)) < 1e-9
