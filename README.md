# ms2deepscore pytorch 2 onnx converter

This library/cli tool aims at converting [ms2deepscore](https://github.com/matchms/ms2deepscore) models from pytorch to [onnx](https://github.com/onnx/onnx).

## Installation
```
git clone git@github.com:julianpollmann/ms2deepscore-onnx-converter.git
cd ms2deepscore-onnx-converter

uv sync
# or using pip:
pip install .
```

## Usage
The tool can either convert a local ms2deepscore model or download one from zenodo and convert it to onnx.
```
# Using the CLI:
ms2ds_onnx https://zenodo.org/records/17826815 -o onnx_model_dir
ms2ds_onnx ms2deepscore_model.pt -o onnx_model_dir

# or within your python script:
from ms2ds_converter import convert_to_onnx

convert_to_onnx("ms2deepscore_model.pt", "onnx_model_dir")
```

## Inference using onnx runtime
After converting a ms2deepscore model to pytorch you can use the [ONNX Runtime](https://onnxruntime.ai/) for inference. You'll need to install onnxruntime  separately.

```
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

    # Load exported SettingsMS2Deepscore.
    with open(
        "onnx_model_dir/ms2deepscore_model_settings.json", "r", encoding="utf-8"
    ) as file:
        settings_dict = json.load(file)

    # Remove spectrum_file_path from settings for inference, since it will fail validation.
    settings_dict["spectrum_file_path"] = None
    settings = SettingsMS2Deepscore(**settings_dict)

    # Load ONNX Model and compute embeddings.
    ort_session = ort.InferenceSession("onnx_model_dir/ms2deepscore_model.onnx")
    embeddings = compute_embeddings_onnx(ort_session, spectra, settings)

    # Use your embeddings in some way...
    np.save("embeddings_onnx.npy", embeddings)


if __name__ == "__main__":
    main()
```



## License
GNU GPLv3. See [License](LICENSE)