import json
import platform

try:
    import openvino  # noqa: F401
except ImportError:
    openvino = None

import numpy as np
import onnxruntime as ort
from matchms import Spectrum
from matchms.importing import load_spectra
from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.tensorize_spectra import tensorize_spectra


def compute_embeddings_onnx(
    onnx_session: ort.InferenceSession,
    spectra: list[Spectrum],
    settings: SettingsMS2Deepscore,
) -> np.ndarray:
    # We use ms2deepscore to create tensors from spectra and convert them to np arrays.
    X_binned_torch, X_metadata_torch = tensorize_spectra(spectra, settings)
    X_binned = X_binned_torch.numpy().astype(np.float32)
    X_metadata = X_metadata_torch.numpy().astype(np.float32)

    # Build the input data, depending on additional metadata in model, e.g. ["input_peaks", "input_metadata"] or just ["input_peaks"].
    input_names = [inp.name for inp in onnx_session.get_inputs()]
    output_name = onnx_session.get_outputs()[0].name

    input_feed = {input_names[0]: X_binned}

    if len(input_names) > 1 and X_metadata.shape[1] > 0:
        input_feed[input_names[1]] = X_metadata

    # inference step.
    embeddings = onnx_session.run([output_name], input_feed)[0]

    return embeddings


def configure_onnx_providers() -> list:
    available = ort.get_available_providers()
    providers = []

    # CUDA
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    # macOS -> use CoreML
    if platform.system() == "Darwin":
        major, minor, *_ = map(int, platform.mac_ver()[0].split("."))
        if (major, minor) >= (12, 0):
            providers.append(
                (
                    "CoreMLExecutionProvider",
                    {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
                )
            )

    # Intel -> OpenVino
    if "OpenVINOExecutionProvider" in available:
        providers.append(("OpenVINOExecutionProvider", {"device_type": "GPU"}))

    # Fallback
    providers.append("CPUExecutionProvider")

    return providers


def main():
    # load some spectra with matchms and maybe do some filtering...
    spectra = list(load_spectra("spectra.mgf"))

    # Load model with attached settings
    providers = configure_onnx_providers()
    ort_session = ort.InferenceSession(
        "../onnx_export/ms2deepscore_model.onnx", providers=providers
    )
    model_metadata = ort_session.get_modelmeta().custom_metadata_map

    if "settings" not in model_metadata:
        raise ValueError(
            "Model does not contain settings. These are required for inference."
        )

    # Remove spectrum_file_path from settings for inference, since it will fail validation.
    settings_dict = json.loads(model_metadata["settings"])
    settings_dict["spectrum_file_path"] = None
    settings = SettingsMS2Deepscore(**settings_dict)

    embeddings = compute_embeddings_onnx(ort_session, spectra, settings)

    # Use your embeddings in some way...
    np.save("embeddings_onnx.npy", embeddings)


if __name__ == "__main__":
    main()
