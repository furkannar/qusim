"""Run every script in examples/ so the tutorial code can never silently break."""

import contextlib
import io
import pathlib
import runpy

import pytest

EXAMPLES = sorted((pathlib.Path(__file__).parent.parent / "examples").glob("*.py"))


def test_examples_exist():
    assert len(EXAMPLES) >= 4


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runpy.run_path(str(script), run_name="__main__")
    assert out.getvalue().strip(), f"{script.name} printed nothing"
