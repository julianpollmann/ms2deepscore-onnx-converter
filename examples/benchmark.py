import json
import time
import numpy as np
import onnxruntime as ort
import platform

try:
    import openvino  # noqa: F401
except ImportError:
    openvino = None

from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.models import compute_embedding_array, load_model
from ms2deepscore.tensorize_spectra import tensorize_spectra
from matchms import Spectrum
from matchms.importing import load_spectra
from matchms.filtering.default_pipelines import (
    CLEAN_PEAKS,
    HARMONIZE_METADATA_FIELD_NAMES,
)
from matchms import SpectrumProcessor
from matchms import filtering as msfilters


NUM_RUNS = 3
SPECTRA_PATH = "../spectra.mgf"
PT_MODEL_PATH = "../onnx_model_dir/zenodo_downloads/ms2deepscore_model.pt"
ONNX_MODEL_PATH = "../onnx_model_dir/ms2deepscore_model.onnx"
ONNX_MODEL_SETTINGS_PATH = "../onnx_model_dir/ms2deepscore_model_settings.json"


spectrum_processor = SpectrumProcessor(
    HARMONIZE_METADATA_FIELD_NAMES
    + CLEAN_PEAKS
    + [
        msfilters.normalize_intensities,
    ]
)


def spectra_importer(file_mgf):
    """Import and basic filtering of spectra from mgf file."""
    spectra = list(load_spectra(file_mgf))

    # For now: simple spectrum processing for more consistency
    spectra, _ = spectrum_processor.process_spectra(spectra)
    cleaned_spectra = [s for s in spectra if s is not None]
    cleaned_spectra2 = [s for s in cleaned_spectra if s.get("inchikey") is not None]

    return cleaned_spectra2


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


def run_benchmark():
    # ----------------------------------------------------
    # 1. Setup
    # ----------------------------------------------------
    print("-> Starting setup: Loading spectra and models...")
    spectra = spectra_importer(SPECTRA_PATH)

    # PyTorch setup
    ms2ds_model = load_model(PT_MODEL_PATH, allow_legacy=True)
    ms2ds_model.eval()

    # ONNX setup
    with open(ONNX_MODEL_SETTINGS_PATH, "r", encoding="utf-8") as file:
        settings_dict = json.load(file)

    providers = configure_onnx_providers()
    settings_dict["spectrum_file_path"] = None
    settings = SettingsMS2Deepscore(**settings_dict)
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

    print(
        f"-> Setup completed. Benchmarking {len(spectra)} spectra over {NUM_RUNS} runs.\n"
    )

    # ----------------------------------------------------
    # 2. BENCHMARK: PYTORCH
    # ----------------------------------------------------
    print("Starting PyTorch benchmark...")
    pytorch_times = []

    for i in range(NUM_RUNS):
        start_time = time.perf_counter()

        # Important: progress_bar=False avoids time-consuming console output
        pt_embeddings = compute_embedding_array(
            ms2ds_model, spectra, progress_bar=False
        )

        end_time = time.perf_counter()

        pytorch_times.append(end_time - start_time)
        print(f"   Run {i + 1}: {pytorch_times[-1]:.4f} seconds")

    # ----------------------------------------------------
    # 3. BENCHMARK: ONNX (BATCH)
    # ----------------------------------------------------
    print("\nStarting ONNX benchmark...")
    onnx_times = []

    for i in range(NUM_RUNS):
        start_time = time.perf_counter()

        onnx_embeddings = compute_embeddings_onnx(ort_session, spectra, settings)

        end_time = time.perf_counter()

        onnx_times.append(end_time - start_time)
        print(f"   Run {i + 1}: {onnx_times[-1]:.4f} seconds")

    # Verify that ONNX and PyTorch produce equivalent embeddings
    assert np.allclose(onnx_embeddings, pt_embeddings, rtol=1e-05, atol=1e-05)

    # ----------------------------------------------------
    # 4. EVALUATION AND METRICS
    # ----------------------------------------------------
    print("\n" + "=" * 45)
    print("              BENCHMARK RESULTS")
    print("=" * 45)

    print("PyTorch inference:")
    print(f"  • Fastest:  {min(pytorch_times):.4f}s")
    print(f"  • Average:  {np.mean(pytorch_times):.4f}s")
    print(f"  • Slowest:  {max(pytorch_times):.4f}s")

    print("\nONNX batch inference:")
    print(f"  • Fastest:  {min(onnx_times):.4f}s")
    print(f"  • Average:  {np.mean(onnx_times):.4f}s")
    print(f"  • Slowest:  {max(onnx_times):.4f}s")

    print("-" * 45)

    # Calculate performance improvement
    speedup = np.mean(pytorch_times) / np.mean(onnx_times)

    print(f"➔ Result: ONNX is {speedup:.2f}x faster on average.")
    print("=" * 45)


if __name__ == "__main__":
    run_benchmark()
