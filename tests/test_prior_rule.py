"""object.s_u_joint: auto, the joint-loss object prior level from the noise and the number of frames."""
from pathlib import Path
import pytest
import torch
import yaml
import torchmfbd

CONFIG = Path(__file__).parent / 'joint.yaml'


def _fit(make_synthetic_frames, s_u, n_frames=5, sigma=0.005):
    with open(CONFIG) as f:
        c = yaml.safe_load(f)
    c['optimization']['gpu'] = -1
    c.pop('object2', None)
    c['object1']['s_u_joint'] = s_u
    obj, frames = make_synthetic_frames(CONFIG, 4, n_frames, sigma, seed=0)
    torch.manual_seed(0)
    d = torchmfbd.Deconvolution(c)
    d.add_frames(frames, sigma=sigma * torch.ones(4, n_frames), id_object=0, id_diversity=0, diversity=0.0)
    d.deconvolve(infer_object=False, optimizer='lbfgs', simultaneous_sequences=200, n_iterations=3)
    return d


def test_auto_equals_the_rule_written_by_hand(make_synthetic_frames):
    auto = _fit(make_synthetic_frames, 'auto')
    # sigma 0.005 given to add_frames (stored in float32: LBFGS turns a difference in the last bit into a visible one)
    mu = float(torch.mean(auto.frames[0]))
    value = 250.0 * (float(torch.mean(0.005 * torch.ones(4, 5))) / mu / 0.02) * (12.0 / 5) * mu**2
    assert auto.s_u_joint_value(0) == pytest.approx(value, rel=1e-6)
    fixed = _fit(make_synthetic_frames, value)
    assert torch.equal(auto.obj[0], fixed.obj[0])
    assert torch.equal(auto.modes_seq[0], fixed.modes_seq[0])


def test_auto_is_scale_free(make_synthetic_frames):
    # the same burst in units 1000 times larger: the level (an object power) scales with the square of the units
    obj, frames = make_synthetic_frames(CONFIG, 4, 5, 0.005, seed=0)
    levels = []
    for k in (1.0, 1000.0):
        with open(CONFIG) as f:
            c = yaml.safe_load(f)
        c['optimization']['gpu'] = -1
        c['object1']['s_u_joint'] = 'auto'
        d = torchmfbd.Deconvolution(c)
        d.add_frames(k * frames, sigma=k * 0.005 * torch.ones(4, 5), id_object=0, id_diversity=0, diversity=0.0)
        levels.append(d.s_u_joint_value(0))
    assert levels[1] == pytest.approx(1e6 * levels[0], rel=1e-4)


def test_number_is_unchanged(make_synthetic_frames):
    d = _fit(make_synthetic_frames, 400.0)
    assert d.s_u_joint_value(0) == 400.0


def test_auto_without_frames_refuses():
    with open(CONFIG) as f:
        c = yaml.safe_load(f)
    c['optimization']['gpu'] = -1
    c['object1']['s_u_joint'] = 'auto'
    d = torchmfbd.Deconvolution(c)
    with pytest.raises(ValueError, match="add_frames"):
        d.s_u_joint_value(0)
