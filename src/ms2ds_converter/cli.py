import argparse
from pathlib import Path
import logging

from ms2ds_converter.converter import convert_to_onnx
from ms2ds_converter.data_loader import (
    download_zenodo_files,
    extract_zenodo_zip_files,
    resolve_model_path,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[ms2ds-converter] %(levelname)s: %(message)s", level=logging.INFO
)


def is_zenodo_identifier(path_or_link: str) -> bool:
    return "zenodo" in path_or_link.lower()


def main():
    parser = argparse.ArgumentParser(
        description="Converts ms2deepscore pytorch model to onnx (supports local files and Zenodo DOIs/URLs)."
    )
    parser.add_argument(
        "model_source",
        type=str,
        help="Path to your local .pt file OR a Zenodo DOI/URL (e.g., 10.5281/zenodo.17826815)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./onnx_export", help="Output dir"
    )

    args = parser.parse_args()
    source = args.model_source
    output_dir = Path(args.output)

    if is_zenodo_identifier(source):
        download_target_dir = output_dir / "zenodo_downloads"
        downloaded_files = download_zenodo_files(
            source, download_dir=download_target_dir
        )
        extract_zenodo_zip_files(downloaded_files, output_path=download_target_dir)
        source = download_target_dir

    try:
        model_file_path = resolve_model_path(source)
    except FileNotFoundError as e:
        logger.error(f"No .pt file found: {e}")
        exit(1)

    convert_to_onnx(model_file_path, args.output)


if __name__ == "__main__":
    main()
