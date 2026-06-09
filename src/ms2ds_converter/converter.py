import json
import torch
import numpy as np
from pathlib import Path
from ms2deepscore.models import load_model
from ms2deepscore import SettingsMS2Deepscore
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[ms2ds-converter] %(levelname)s: %(message)s", level=logging.INFO
)


def get_metadata_length(model_settings: SettingsMS2Deepscore) -> int:
    """Derive the Metadata length from the provided pytorch model settings.

    Parameters
    ----------
    model_settings : SettingsMS2Deepscore
        SettingsMS2Deepscore of the pytorch model.

    Returns
    -------
    int
        number of metadata in settings.
    """
    if (
        hasattr(model_settings, "additional_metadata")
        and model_settings.additional_metadata
    ):
        return len(model_settings.additional_metadata)

    return 0


def convert_to_onnx(pytorch_model_path: Path, output_dir: Path):
    """Converts a ms2deepscore pytorch model to onnx.

    Parameters
    ----------
    pytorch_model_path : Path
        The full path to the .pt model file.
    output_dir : Path
        The output dir to store the onnx model.
    """
    try:
        model = load_model(pytorch_model_path)
    except (ValueError, RuntimeError, TypeError):
        logger.warning(
            "Model contains unsafe tensors. It was loaded using legacy weights."
        )
        model = load_model(pytorch_model_path, allow_legacy=True)

    encoder = model.encoder
    encoder.eval()

    # Get shape of input features
    first_linear_layer = encoder.dense_layers[0][0]
    total_in_features = first_linear_layer.in_features

    # Get shape of metadata
    num_metadata = get_metadata_length(model.model_settings)

    # Get bins
    num_peaks = total_in_features - num_metadata

    # Dummy inputs
    dummy_peaks = torch.randn(1, num_peaks)
    logger.info(f"Will use {dummy_peaks.shape} as dummy_peaks")

    if num_metadata > 0:
        dummy_meta = torch.randn(1, num_metadata)
        dummy_inputs = (dummy_peaks, dummy_meta)
        input_names = ["input_peaks", "input_metadata"]
        dynamic_shapes = {
            "spectra_tensors": ["batch_size", None],
            "metadata_tensors": ["batch_size", None],
        }
    else:
        dummy_inputs = (dummy_peaks,)
        input_names = ["input_peaks"]
        dynamic_shapes = {
            "spectra_tensors": ["batch_size", None],
        }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_name = Path(pytorch_model_path).stem
    onnx_file = Path(out_path, model_name).with_suffix(".onnx")
    json_file = Path(out_path, f"{model_name}_settings").with_suffix(".json")

    logger.info(f"Export ONNX model to {out_path}")

    torch.onnx.export(
        encoder,
        dummy_inputs,
        str(onnx_file),
        export_params=True,
        opset_version=None,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["embedding"],
        dynamic_shapes=dynamic_shapes,
    )

    # Convert model settings to json
    settings_dict = vars(model.model_settings).copy()
    for key, value in settings_dict.items():
        if isinstance(value, np.ndarray):
            settings_dict[key] = value.tolist()

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=4)

    logger.info("Conversion successful.")
