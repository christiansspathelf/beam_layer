import pytest

from beam_layer import (
    BeamParameters,
    BeamSectionForces,
    ConcreteMaterial,
    LayeredBeamSection,
    RebarFace,
    SteelMaterial,
    sweep_curvature,
    sweep_load_factor,
)


@pytest.fixture
def params():
    return BeamParameters(
        height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0,
        bottom=RebarFace(3, 16.0), top=RebarFace(2, 12.0),
    )


@pytest.fixture
def concrete():
    return ConcreteMaterial(fck=30.0)


@pytest.fixture
def steel():
    return SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.05)


def test_sweep_stays_within_capacity_returns_all_points_no_diagnosis(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    path, diag = sweep_load_factor(section, direction, n_points=6, max_load_factor=10.0)

    assert len(path.points) == 6
    assert path.points[0].lam == pytest.approx(0.0)
    assert path.points[-1].lam == pytest.approx(10.0)
    assert all(p.converged for p in path.points)
    assert diag is None


def test_sweep_stops_at_a_genuine_capacity_limit(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=30)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=100.0)
    path, diag = sweep_load_factor(section, direction, n_points=8, max_load_factor=2.0)

    assert len(path.points) < 8
    assert not path.points[-1].converged
    assert diag is not None
    assert diag.verdict == "likely_physical_limit"


def test_n_points_below_two_raises(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    with pytest.raises(ValueError):
        sweep_load_factor(section, direction, n_points=1, max_load_factor=1.0)


def test_curvature_sweep_stays_within_capacity_returns_all_points_no_diagnosis(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    path, diag = sweep_curvature(section, direction, step=0.002, ramp_substeps=3, max_points=6)

    assert len(path.points) == 6
    assert path.points[0].kappa == pytest.approx(0.0)
    assert all(p.converged for p in path.points)
    assert diag is None


def test_curvature_sweep_ramps_finer_before_settling_into_the_regular_step(params, concrete, steel):
    """The first `ramp_substeps` points sample the first `step` of
    curvature at finer resolution (to resolve the uncracked/linear region
    near kappa=0, which can be much shorter than one coarse step); every
    point after that adds a full `step`."""
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    path, _ = sweep_curvature(section, direction, step=0.002, ramp_substeps=5, max_points=12)

    kappas = [p.kappa for p in path.points]
    ramp_steps = [kappas[i + 1] - kappas[i] for i in range(5)]
    later_steps = [kappas[i + 1] - kappas[i] for i in range(5, len(kappas) - 1)]
    assert max(ramp_steps) == pytest.approx(0.002 / 5)
    assert all(s == pytest.approx(0.002 / 5) for s in ramp_steps)
    assert all(s == pytest.approx(0.002) for s in later_steps)
    assert kappas[5] == pytest.approx(0.002)  # the ramp's last point lands exactly on one full step


def test_curvature_sweep_stops_at_a_genuine_capacity_limit(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=30)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    path, diag = sweep_curvature(section, direction, step=0.004, max_points=40)

    assert len(path.points) < 40
    assert diag is not None
    assert diag.verdict == "likely_physical_limit"
    assert diag.concrete_crushed


def test_curvature_sweep_stops_even_if_the_final_point_still_converges(params, concrete, steel):
    """Regression guard: solve_at_curvature can keep converging to a
    lower-moment equilibrium well past first crushing (concrete crushes
    layer-by-layer, not all at once - see load_sweep.py's module
    docstring), so the sweep must stop at the model's own hard crushing
    criterion even on a nominally-converged point, not only on a failed
    solve."""
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=30)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    # max_points well past where crushing actually starts (~0.08, per the
    # sibling test), so the sweep has room to demonstrate it stops early
    # rather than happening to land its last scheduled point there.
    path, diag = sweep_curvature(section, direction, step=0.008, max_points=30)

    assert diag is not None
    assert diag.verdict == "likely_physical_limit"
    assert len(path.points) < 30  # stopped before exhausting max_points
    # the sweep must stop at the *first* point that trips the hard
    # criteria - i.e. the curve's own running peak moment is at most one
    # step behind the point the sweep actually stopped at, not several
    # points (which is what happened before _hard_criteria_tripped was
    # added: the solve kept "converging" to a falling moment for many
    # more steps past the true peak).
    m_values = [p.forces[1] for p in path.points]
    peak_index = m_values.index(max(m_values))
    assert peak_index >= len(m_values) - 2


def test_curvature_sweep_maintains_proportional_loading_direction(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=-50.0, v_z=0, m_y=20.0)
    path, _ = sweep_curvature(section, direction, step=0.002, ramp_substeps=3, max_points=8)

    target_ratio = direction.n_x / direction.m_y
    for p in path.points[1:]:  # skip the trivial kappa=0 point (0/0)
        if p.converged:
            assert p.forces[0] / p.forces[1] == pytest.approx(target_ratio, rel=1e-3)


def test_curvature_sweep_hogging_direction_gives_negative_curvature(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=-1.0)
    path, _ = sweep_curvature(section, direction, step=0.002, ramp_substeps=3, max_points=6)

    assert path.points[-1].kappa < 0
    assert path.points[-1].forces[1] < 0


def test_curvature_sweep_non_positive_step_raises(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    with pytest.raises(ValueError):
        sweep_curvature(section, direction, step=0.0)


def test_curvature_sweep_max_points_too_small_for_ramp_raises(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    with pytest.raises(ValueError):
        sweep_curvature(section, direction, step=0.002, ramp_substeps=10, max_points=5)


def test_curvature_sweep_reaching_max_points_returns_no_diagnosis(params, concrete, steel):
    """A section that's still nowhere near capacity when the safety cap
    is hit should behave like sweep_load_factor exhausting max_load_factor
    without finding a limit: all points returned, no FailureDiagnosis."""
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    path, diag = sweep_curvature(section, direction, step=0.0001, ramp_substeps=3, max_points=6)

    assert len(path.points) == 6
    assert diag is None
