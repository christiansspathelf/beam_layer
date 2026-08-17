import pytest

from shell_layer import (
    ConcreteMaterial,
    LayeredSection,
    ReinforcementLayout,
    SectionForces,
    ShellParameters,
    SteelMaterial,
    sweep_load_factor,
)


@pytest.fixture
def params():
    return ShellParameters(
        thickness=0.250, cover=0.030,
        top_x=ReinforcementLayout(bar_diameter=12.0, spacing=150.0),
        top_y=ReinforcementLayout(bar_diameter=12.0, spacing=150.0),
        bottom_x=ReinforcementLayout(bar_diameter=16.0, spacing=150.0),
        bottom_y=ReinforcementLayout(bar_diameter=12.0, spacing=150.0),
    )


@pytest.fixture
def concrete():
    return ConcreteMaterial(fck=40.0, eps_c0=0.00188)


@pytest.fixture
def steel():
    return SteelMaterial(fy=500.0, fu=625.0, Es=205_000.0, eps_su=0.10)


def test_sweep_stays_within_capacity_returns_all_points_no_diagnosis(params, concrete, steel):
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=20)
    direction = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=1.0)
    path, diag = sweep_load_factor(section, direction, n_points=6, max_load_factor=20.0)

    assert len(path.points) == 6
    assert path.points[0].lam == pytest.approx(0.0)
    assert path.points[-1].lam == pytest.approx(20.0)
    assert all(p.converged for p in path.points)
    assert diag is None


def test_sweep_stops_at_a_genuine_capacity_limit(params, concrete, steel):
    """Case 2 pure torsion (see shell-layer-torsion-solver-investigation
    project memory) has a real, independently-confirmed ~76.3 kNm/m
    capacity limit - a sweep pushed well past that should stop early with
    a physical-limit diagnosis rather than grinding out every point.
    """
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=40)
    direction = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=100.0)
    # max_load_factor=1.5 -> m_xy up to 150, well past the ~76-85 kNm/m
    # limit, with several scheduled points beyond it - a genuine early
    # stop, not just the sweep's last scheduled point happening to fail.
    path, diag = sweep_load_factor(section, direction, n_points=8, max_load_factor=1.5)

    assert len(path.points) < 8
    assert not path.points[-1].converged
    assert diag is not None
    assert diag.verdict == "likely_physical_limit"


def test_n_points_below_two_raises(params, concrete, steel):
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=20)
    direction = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=1.0)
    with pytest.raises(ValueError):
        sweep_load_factor(section, direction, n_points=1, max_load_factor=1.0)
