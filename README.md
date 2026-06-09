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

## License
GNU GPLv3. See [License](LICENSE)