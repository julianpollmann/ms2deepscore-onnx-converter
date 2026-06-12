from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from ms2deepscore import SettingsMS2Deepscore
from ms2ds_converter.converter import convert_to_onnx, get_metadata_length
import logging

@pytest.fixture
def model_path():
    return Path(__file__).parent / "ms2deepscore_model.pt"


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=SettingsMS2Deepscore)
    settings.additional_metadata = ["precursor_mz", "charge"]
    settings.embedding_dim = 500
    settings.intensity_scaling = 0.5
    settings.min_mz = 10
    settings.max_mz = 1000
    settings.mz_bin_width = 0.1
    settings.number_of_bins = 9900
    return settings


@pytest.fixture
def mock_model(mock_settings):
    model = MagicMock()
    model.model_settings = mock_settings
    mock_layer = MagicMock()
    mock_layer.in_features = 100
    model.encoder.dense_layers = [[mock_layer]]
    return model


def test_get_metadata_length_with_metadata(mock_settings):
    length = get_metadata_length(mock_settings)
    assert length == 2


def test_get_metadata_length_no_metadata():
    settings = MagicMock(spec=SettingsMS2Deepscore)
    settings.additional_metadata = []
    assert get_metadata_length(settings) == 0

    del settings.additional_metadata
    assert get_metadata_length(settings) == 0


def test_convert_to_onnx_success_with_metadata(tmp_path, model_path):
    output_dir = str(tmp_path)

    convert_to_onnx(str(model_path), output_dir)

    assert (tmp_path / "ms2deepscore_model.onnx").exists()
    assert (tmp_path / "ms2deepscore_model_settings.json").exists()


@patch("ms2ds_converter.converter.load_model")
@patch("ms2ds_converter.converter.torch.onnx.export")
def test_convert_to_onnx_success_no_metadata(
    mock_torch_export, mock_load_model, tmp_path, mock_model
):
    mock_model.model_settings.additional_metadata = []
    mock_load_model.return_value = mock_model

    with patch("builtins.open"), patch("json.dump"):
        convert_to_onnx("dummy_model.pt", str(tmp_path))
        _, kwargs = mock_torch_export.call_args
        assert kwargs["input_names"] == ["input_peaks"]
        mock_model.model_settings.save_to_file.assert_called_once()


@patch("ms2ds_converter.converter.load_model")
@patch("ms2ds_converter.converter.torch.onnx.export")
def test_convert_to_onnx_fallback_to_legacy(
    mock_torch_export, mock_load_model, tmp_path, mock_model
):
    mock_load_model.side_effect = [ValueError("Unsafe tensors"), mock_model]

    with patch("builtins.open"), patch("json.dump"):
        convert_to_onnx("dummy_model.pt", str(tmp_path))
        assert mock_load_model.call_count == 2


@patch("ms2ds_converter.converter.load_model")
@patch("ms2ds_converter.converter.torch.onnx.export")
def test_convert_to_onnx_required_keys_present(
    mock_torch_export, mock_load_model, tmp_path, mock_model, caplog
):
    """Testet, dass kein Error geloggt wird, wenn alle required_keys vorhanden sind."""
    mock_load_model.return_value = mock_model

    with caplog.at_level(logging.ERROR):
        convert_to_onnx("dummy_model.pt", str(tmp_path))

    errors = [r.message for r in caplog.records if "do not contain required attribute" in r.message]
    assert len(errors) == 0


@patch("ms2ds_converter.converter.load_model")
@patch("ms2ds_converter.converter.torch.onnx.export")
def test_convert_to_onnx_missing_required_keys(
    mock_torch_export, mock_load_model, tmp_path, mock_model, caplog
):
    """Testet, dass spezifische ERROR-Logs geschrieben werden, wenn required_keys fehlen."""
    mock_load_model.return_value = mock_model
    
    del mock_model.model_settings.min_mz
    del mock_model.model_settings.embedding_dim

    with caplog.at_level(logging.ERROR):
        convert_to_onnx("dummy_model.pt", str(tmp_path))

    assert any("do not contain required attribute min_mz" in r.message for r in caplog.records)
    assert any("do not contain required attribute embedding_dim" in r.message for r in caplog.records)
    
    assert not any("do not contain required attribute max_mz" in r.message for r in caplog.records)