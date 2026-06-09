import logging
import re
from pathlib import Path
from typing import List
import pooch
import requests
from zipfile import ZipFile

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[ms2ds-converter] %(levelname)s: %(message)s", level=logging.INFO
)


def _extract_record_id(zenodo_identifier: str) -> str:
    doi_match = re.search(r"10\.5281/zenodo\.(\d+)", zenodo_identifier)
    if doi_match:
        return doi_match.group(1)

    url_match = re.search(r"zenodo\.org/(?:record|records)/(\d+)", zenodo_identifier)
    if url_match:
        return url_match.group(1)

    raise ValueError(f"Invalid Zenodo-Identifier: {zenodo_identifier}")


def download_zenodo_files(
    zenodo_identifier: str, download_dir: str | Path = "./downloaded_models"
) -> List[Path]:
    """Downloads most recent files for a zenodo identifier.

    Parameters
    ----------
    zenodo_identifier : str
        The Zenodo Identifier to download files from.
    download_dir : str | Path
        The directory to download files to. Defaults to downloaded_models.

    Raises
    ------
    RuntimeError if Zenodo Record has no associated files.
    """
    record_id = _extract_record_id(zenodo_identifier)
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url, timeout=15)
    response.raise_for_status()
    data = response.json()

    # Get the most recent version, if not already latest
    if "links" in data and "latest" in data["links"]:
        latest_api_url = data["links"]["latest"]
        if api_url != latest_api_url:
            response = requests.get(latest_api_url, timeout=15)
            response.raise_for_status()
            data = response.json()

    files = data.get("files", [])
    if not files:
        raise RuntimeError(f"Zenodo Record {record_id} has no files associated.")

    downloaded_files = []

    for file_info in files:
        file_name = Path(file_info["key"])
        download_url = file_info["links"]["self"]

        # ZEnodo file checksum is a md5:...
        checksum = file_info.get("checksum", "")

        file_path = pooch.retrieve(
            url=download_url,
            known_hash=checksum,
            path=download_path,
            fname=file_name.name,
            progressbar=True,
        )

        downloaded_files.append(Path(file_path))

    return downloaded_files


def extract_zenodo_zip_files(file_paths: List[Path], output_path: str | Path = "./"):
    for file in file_paths:
        if file.suffix in [".zip"]:
            with ZipFile(file, "r") as zip_file:
                zip_file.extractall(output_path)


def resolve_model_path(
    source_path: str | Path, default_filename: str = "ms2deepscore_model.pt"
) -> Path:
    """Checks if source_path is a .pt file or directory. Will assume ms2deepscore_model.pt exists if source_path is directory.

    Parameters
    ----------
    source_path : Path
        The full path to the .pt model  or a directory.
    default_filename : Path
        default_filename if source_path is a directory. Defaults to ms2deepscore_model.pt.

    Raises
    ------
    FileNotFoundError if not .pt file is present.
    """
    path = Path(source_path)

    if path.is_file() and path.suffix == ".pt":
        return path

    if path.is_dir():
        target_file = path / default_filename
        if target_file.is_file():
            return target_file

    raise FileNotFoundError(f"No valid .pt file in {source_path}.")
