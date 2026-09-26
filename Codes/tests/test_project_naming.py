"""Current branding must not change compatibility or scientific identities."""
from pathlib import Path

import balds
from balds.schema.identity import PROTOCOL_VERSION


def test_current_distribution_name():
    metadata = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert '\nname = "snr-lds"\n' in metadata
    assert 'name = "ba-lds"' not in metadata
    assert "BA-LDS paper" not in metadata
    assert "SNR-LDS" in balds.__doc__


def test_compatibility_entry_points_and_rng_identity():
    metadata = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    for name, target in (
        ("balds", "balds.cli.ba:main"),
        ("balds-run", "balds.cli.main:main"),
        ("balds-repeat", "balds.cli.repeatability:main"),
    ):
        assert f'{name} = "{target}"' in metadata
    # This is a historical RNG/hash domain, not the release's display name.
    assert PROTOCOL_VERSION == "CFA-PRED-PILOT-v1"
