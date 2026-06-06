import logging
import subprocess
import sys

logging.basicConfig(
    format="[ms2ds-converter] %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def test_cli_help():
    logger.info("Starting smoke test for the CLI tool...")
    try:
        result = subprocess.run(
            ["ms2ds_onnx", "--help"], capture_output=True, text=True, check=True
        )

        if "usage:" in result.stdout.lower() or "options:" in result.stdout.lower():
            logger.info("Smoke test successful: CLI responded correctly.")
        else:
            logger.warning(
                "CLI started successfully, but the output was unexpected:"
            )
            print(result.stdout)

    except subprocess.CalledProcessError as e:
        logger.error(
            f"Smoke test failed! CLI crashed with exit code {e.returncode}"
        )
        logger.error(f"Error output: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(
            "Smoke test failed! The command 'ms2ds_onnx' was not found in the environment."
        )
        sys.exit(1)


if __name__ == "__main__":
    test_cli_help()