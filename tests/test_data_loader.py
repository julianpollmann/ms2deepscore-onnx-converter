import pytest
from unittest.mock import patch, MagicMock

from ms2ds_converter.data_loader import _extract_record_id, resolve_model_path, extract_zenodo_zip_files, download_zenodo_files

@pytest.mark.parametrize(
    "identifier, expected_id",
    [
        ("10.5281/zenodo.17826815", "17826815"),
        ("https://zenodo.org/records/17826815", "17826815"),
        ("https://zenodo.org/record/123456", "123456"),
    ],
)
def test_extract_record_id_success(identifier, expected_id):
    assert _extract_record_id(identifier) == expected_id


def test_extract_record_id_invalid():
    with pytest.raises(ValueError, match="Invalid Zenodo-Identifier"):
        _extract_record_id("invalid-identifier-123")


def test_resolve_model_path_with_direct_file(tmp_path):
    model_file = tmp_path / "my_model.pt"
    model_file.touch()

    result = resolve_model_path(model_file)
    assert result == model_file


def test_resolve_model_path_with_directory(tmp_path):
    default_file = tmp_path / "ms2deepscore_model.pt"
    default_file.touch()

    result = resolve_model_path(tmp_path)
    assert result == default_file


def test_resolve_model_path_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No valid .pt file"):
        resolve_model_path(tmp_path)


def test_extract_zenodo_zip_files(tmp_path):
    import zipfile

    zip_path = tmp_path / "test.zip"
    content_file = tmp_path / "dummy.txt"
    content_file.write_text("hello")

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(content_file, arcname="dummy.txt")

    output_dir = tmp_path / "extracted"
    output_dir.mkdir()

    extract_zenodo_zip_files([zip_path], output_path=output_dir)

    assert (output_dir / "dummy.txt").exists()
    assert (output_dir / "dummy.txt").read_text() == "hello"


@patch("ms2ds_converter.data_loader.requests.get")
@patch("ms2ds_converter.data_loader.pooch.retrieve")
def test_download_zenodo_files_success(
    mock_pooch_retrieve, mock_requests_get, tmp_path
):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "links": {"latest": "https://zenodo.org/api/records/12345"},
        "files": [
            {
                "key": "model.pt",
                "links": {"self": "https://zenodo.org/api/files/xyz"},
                "checksum": "md5:1234567890abcdef",
            }
        ],
    }
    mock_requests_get.return_value = mock_response

    expected_downloaded_path = tmp_path / "model.pt"
    mock_pooch_retrieve.return_value = str(expected_downloaded_path)

    result = download_zenodo_files("10.5281/zenodo.12345", download_dir=tmp_path)

    assert len(result) == 1
    assert result[0] == expected_downloaded_path
    mock_requests_get.assert_called_with(
        "https://zenodo.org/api/records/12345", timeout=15
    )
    mock_pooch_retrieve.assert_called_once()


@patch("ms2ds_converter.data_loader.requests.get")
def test_download_zenodo_files_no_files_error(mock_requests_get, tmp_path):
    mock_response = MagicMock()
    mock_response.json.return_value = {"files": []}
    mock_requests_get.return_value = mock_response

    with pytest.raises(RuntimeError, match="has no files associated"):
        download_zenodo_files("10.5281/zenodo.12345", download_dir=tmp_path)