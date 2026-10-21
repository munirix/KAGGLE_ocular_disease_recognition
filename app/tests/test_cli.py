"""test_cli.py -- testes unitarios do parser da CLI."""
from __future__ import annotations

import pytest

from main import _build_parser


def _parse(args):
    return _build_parser().parse_args(args)


def test_parser_all_flags():
    args = _parse(["all", "--max-per-class", "10", "--force", "--seed", "123",
                   "--jobs", "2", "--no-tune", "--target-sensitivity", "0.85"])
    assert args.command == "all"
    assert args.max_per_class == 10
    assert args.force is True
    assert args.seed == 123
    assert args.jobs == 2
    assert args.no_tune is True
    assert args.target_sensitivity == 0.85


def test_parser_predict_defaults():
    args = _parse(["predict", "img.jpg"])
    assert args.command == "predict"
    assert args.image == "img.jpg"
    assert args.stages == [2, 3, 4]
    assert args.output is None


def test_parser_predict_custom():
    args = _parse(["predict", "img.jpg", "--stages", "2", "--output", "out.json"])
    assert args.stages == [2]
    assert args.output.name == "out.json"


@pytest.mark.parametrize("cmd", ["eda", "stage1", "stage2", "stage3", "stage4", "ablation", "all"])
def test_parser_pipeline_commands(cmd):
    args = _parse([cmd])
    assert args.command == cmd