"""Install a pinned whisper.cpp CLI and its multilingual Turbo model for this checkout."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

VOICE_DIR = Path(__file__).resolve().parents[1] / ".voice"
SOURCE_URL = "https://github.com/ggml-org/whisper.cpp.git"
SOURCE_COMMIT = "4979e04f5dcaccb36057e059bbaed8a2f5288315"  # v1.8.2
MODEL_NAME = "ggml-large-v3-turbo-q5_0.bin"
MODEL_REV = "5359861c739e955e79d9a303bcbc70fb988958b1"
MODEL_URL = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{MODEL_REV}/{MODEL_NAME}"
MODEL_SHA256 = "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"


def matches_model(path: Path) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == MODEL_SHA256


def install_model(work: Path, local_model: Path | None) -> None:
    model = VOICE_DIR / "models" / MODEL_NAME
    if matches_model(model):
        return
    staged = work / MODEL_NAME
    if local_model is None:
        subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--connect-timeout",
                        "15", "--output", str(staged), MODEL_URL], check=True)
    else:
        shutil.copyfile(local_model, staged)
    if not matches_model(staged):
        raise ValueError(f"{MODEL_NAME} failed SHA-256 verification. Download the pinned model again.")
    model.parent.mkdir(exist_ok=True)
    staged.replace(model)


def install_cli(work: Path) -> None:
    source = work / "whisper.cpp"
    build = work / "build"
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", SOURCE_URL, SOURCE_COMMIT], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--quiet", "--detach", SOURCE_COMMIT], check=True)
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=OFF", "-DWHISPER_BUILD_EXAMPLES=ON",
                    "-DWHISPER_BUILD_TESTS=OFF", "-DWHISPER_BUILD_SERVER=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--target",
                    "whisper-cli", "--parallel", "4"], check=True)
    name = "whisper-cli.exe" if os.name == "nt" else "whisper-cli"
    binary = build / "bin" / name
    if not binary.is_file():  # multi-configuration generators (e.g. Visual Studio)
        binary = build / "bin" / "Release" / name
    subprocess.run([str(binary), "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    bin_dir = VOICE_DIR / "bin"
    bin_dir.mkdir(exist_ok=True)
    binary.replace(bin_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, help=f"reuse a downloaded {MODEL_NAME}")
    args = parser.parse_args()
    programs = ["git", "cmake"]
    if not args.model_file and not (VOICE_DIR / "models" / MODEL_NAME).is_file():
        programs.append("curl")
    missing = [p for p in programs if not shutil.which(p)]
    if missing:
        parser.exit(1, f"Install the missing build tools and retry: {', '.join(missing)}\n")
    try:
        VOICE_DIR.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=VOICE_DIR, prefix="setup-") as tmp:
            work = Path(tmp)
            install_model(work, args.model_file)
            install_cli(work)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Voice setup failed: {exc}\n")
    print(f"Local speech recognition installed in {VOICE_DIR}. Reload the navigator to use it.")


if __name__ == "__main__":
    main()
