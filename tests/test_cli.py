import sys
from pathlib import Path
from unittest.mock import patch
import pytest

from ms2ds_converter.cli import main as cli_main


@pytest.fixture
def model_path():
    return Path(__file__).parent / "ms2deepscore_model.pt"


@patch("ms2ds_converter.cli.convert_to_onnx")
def test_cli_default_output(mock_convert, model_path):
    test_args = ["cli.py", str(model_path)]
    with patch.object(sys, "argv", test_args):
        cli_main()
        mock_convert.assert_called_once_with(Path(model_path), "./onnx_export")


@patch("ms2ds_converter.cli.convert_to_onnx")
def test_cli_custom_output(mock_convert, model_path):
    test_args = ["cli.py", str(model_path), "-o", "/my/custom/path"]
    with patch.object(sys, "argv", test_args):
        cli_main()
        mock_convert.assert_called_once_with(Path(model_path), "/my/custom/path")


def test_cli_missing_argument():
    test_args = ["cli.py"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit):
            cli_main()
