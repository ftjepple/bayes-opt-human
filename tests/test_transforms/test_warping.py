"""Tests for input warping."""

import pytest

from bayesopt_human.transforms.warping import InputWarping


class TestInputWarping:
    def test_sufficient_data(self):
        assert InputWarping.sufficient_data(30, 3) is True
        assert InputWarping.sufficient_data(29, 3) is False
        assert InputWarping.sufficient_data(10, 1) is True

    def test_create_warp_transform(self):
        iw = InputWarping(n_dims=3)
        warp = iw.create_warp_transform()
        assert warp is not None

    def test_get_learned_parameters(self):
        iw = InputWarping(n_dims=2)
        warp = iw.create_warp_transform()
        params = InputWarping.get_learned_parameters(warp)
        assert "concentration0" in params
        assert "concentration1" in params
        assert len(params["concentration0"]) == 2
