import json

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


def main():
    # load some spectra with matchms and maybe do some filtering...
    spectra = list(load_spectra("spectra.mgf"))

    print(ort.get_available_providers())

    # Load exported SettingsMS2Deepscore.
    with open(
        "../onnx_model_dir/ms2deepscore_model_settings.json", "r", encoding="utf-8"
    ) as file:
        settings_dict = json.load(file)

    # Remove spectrum_file_path from settings for inference, since it will fail validation.
    settings_dict["spectrum_file_path"] = None
    settings = SettingsMS2Deepscore(**settings_dict)

    # Load ONNX Model and compute embeddings.
    # ort_session = ort.InferenceSession("onnx_model_dir/ms2deepscore_model.onnx")
    ort_session = ort.InferenceSession(
        "../onnx_model_dir/ms2deepscore_model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    embeddings = compute_embeddings_onnx(ort_session, spectra, settings)

    # Use your embeddings in some way...
    np.save("embeddings_onnx.npy", embeddings)


if __name__ == "__main__":
    main()
