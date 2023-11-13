import numpy as np
import pytest
from pytest import approx

from ste_NLDL.classes.config import Config
from ste_NLDL.classes.observationmodel import VergassolaPlumeModel as ObservationModel

"""
Unit tests for class observationmodel
"""

def setup_1D():
    cfg = Config("test_ste/evaluate_infotaxis_1D_config.yml")
    mod = ObservationModel(cfg)
    return mod


def setup_2D():
    cfg = Config("test_ste/evaluate_infotaxis_2D_config.yml")
    mod = ObservationModel(cfg)
    return mod


def test_autoset_Nhits():
    mod = setup_2D()
    assert mod.Ndim == 2
    assert mod.lambda_over_dx == approx(3162.2776601683795)
    assert np.all(mod.flux_range == np.array([1]))
    assert mod.Nhits == 3


def test_get_mu():
    mod = setup_2D()
    # higher flux -> higher mu
    assert mod._get_mu(flux=2, d_over_dx=1, d_over_dx_downwind=1) > mod._get_mu(flux=1, d_over_dx=1, d_over_dx_downwind=1)
    # closer to flux -> higher mu
    assert (
        mod._get_mu(flux=1, d_over_dx=0, d_over_dx_downwind=1)
        > mod._get_mu(flux=1, d_over_dx=1, d_over_dx_downwind=1)
        > mod._get_mu(flux=1, d_over_dx=2, d_over_dx_downwind=1)
    )


def test_Poisson():
    mod = setup_2D()
    mu = mod._get_mu(flux=1, d_over_dx=0, d_over_dx_downwind=1)
    assert (
        mod._Poisson(mu, hit=0) + mod._Poisson(mu, hit=1) + mod._Poisson(mu, hit=2)
        < 1.0
    )


def test_hit_probabilities():
    mod = setup_2D()
    hit_probabilities = mod._get_hit_probabilities(agent=np.array([0,0]), source=np.array([0,0]), flux=1)
    assert hit_probabilities.shape == (mod.Nhits,)
    assert np.sum(hit_probabilities) == 1


def test_get_observation():
    mod = setup_2D()
    agent = np.array([0, 0])
    source = np.array([0, 0])
    observation = mod._get_observation(agent, source, flux=1, flux_index=0, seed=None)
    assert observation == mod.Nhits - 1

def test_get_mu_wind():
    mod = setup_2D()
    mu = mod._get_mu(flux=1, d_over_dx=1, d_over_dx_downwind=1)
    mod_wind = setup_2D()
    mod_wind.V = 0.00000001
    mod_wind._autoset_Nhits()
    mu_wind = mod_wind._get_mu(flux=1, d_over_dx=1, d_over_dx_downwind=1)
    assert mu == approx(mu_wind)

def test_hit_probabilities_meshgrid_1D():
    mod = setup_1D()
    hit_probabilities_meshgrid = mod._get_hit_probabilities_meshgrid()
    assert mod.Ndim == 1
    assert mod.Ngrid == 3
    assert mod.Nflux == 1
    assert mod.Nhits == 5
    assert hit_probabilities_meshgrid.shape == (mod.Nhits, mod.Nflux, 2 * mod.Ngrid + 1)
    # sum of probabilities should be 1.0
    assert np.all(np.sum(hit_probabilities_meshgrid, axis=0) == 1.0) 
    # probability for hit < Nhits at origin should be 0.0
    assert (
        hit_probabilities_meshgrid[0, 0, mod.Ngrid]
        == hit_probabilities_meshgrid[1, 0, mod.Ngrid]
        == hit_probabilities_meshgrid[2, 0, mod.Ngrid]
        == hit_probabilities_meshgrid[3, 0, mod.Ngrid]
        == 0.0
    )
    # probability for hit = Nhits at origin should be 1.0
    assert hit_probabilities_meshgrid[4, 0, mod.Ngrid] == 1.0


def test_hit_probabilities_meshgrid_2D():
    mod = setup_2D()
    hit_probabilities_meshgrid = mod._get_hit_probabilities_meshgrid()
    assert mod.Nhits == 3
    assert mod.Ndim == 2
    assert mod.Ngrid == 5
    assert mod.Nflux == 1
    assert hit_probabilities_meshgrid.shape == (mod.Nhits, mod.Nflux, 2 * mod.Ngrid + 1, 2 * mod.Ngrid + 1)
    # sum of probabilities should be 1.0
    assert np.all(np.sum(hit_probabilities_meshgrid, axis=0) == 1.0) 
    # probability for hit < Nhits at origin should be 0.0
    assert (
        hit_probabilities_meshgrid[0, 0, mod.Ngrid, mod.Ngrid]
        == hit_probabilities_meshgrid[1, 0, mod.Ngrid, mod.Ngrid]
        == 0.0
    )
    assert hit_probabilities_meshgrid[2, 0, mod.Ngrid, mod.Ngrid] == 1.0