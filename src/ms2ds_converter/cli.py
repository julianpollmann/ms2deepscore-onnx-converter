import argparse
from ms2ds_converter.converter import convert_to_onnx


def main():
    parser = argparse.ArgumentParser(
        description="Converts ms2deepscore pytorch model to onnx."
    )
    parser.add_argument("model_path", type=str, help="Path to your .pt file")
    parser.add_argument(
        "-o", "--output", type=str, default="./onnx_export", help="Output dir"
    )

    args = parser.parse_args()

    convert_to_onnx(args.model_path, args.output)


if __name__ == "__main__":
    main()
