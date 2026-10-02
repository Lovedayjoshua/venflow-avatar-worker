import base64
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod


APP_DIR = Path("/app/MuseTalk")
INPUT_DIR = Path(os.getenv("VENFLOW_INPUT_DIR", "/workspace/input"))
OUTPUT_DIR = Path(os.getenv("VENFLOW_OUTPUT_DIR", "/workspace/output"))
AVATAR_CACHE = Path(os.getenv("VENFLOW_AVATAR_CACHE", "/workspace/avatar-cache"))

VERSION = os.getenv("MUSETALK_VERSION", "v15")

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AVATAR_CACHE.mkdir(parents=True, exist_ok=True)


def error(message, code="WORKER_ERROR", details=None):
    result = {
        "status": "failed",
        "error": {
            "code": code,
            "message": message,
        },
    }

    if details:
        result["error"]["details"] = details

    return result


def download_file(url: str, destination: Path):
    response = requests.get(
        url,
        stream=True,
        timeout=(30, 600),
        headers={"User-Agent": "Venflow-Avatar-Worker/1.0"},
    )

    response.raise_for_status()

    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)

    return destination


def write_base64(data: str, destination: Path):
    if "," in data and data.startswith("data:"):
        data = data.split(",", 1)[1]

    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as output:
        output.write(base64.b64decode(data))

    return destination


def get_input_file(value, destination: Path):
    if not value:
        raise ValueError("Input file is required.")

    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            return download_file(value, destination)

        if value.startswith("data:"):
            return write_base64(value, destination)

        local_path = Path(value)

        if local_path.exists():
            shutil.copy2(local_path, destination)
            return destination

    raise ValueError(
        "Input must be an HTTP(S) URL, data URL, or existing local file."
    )


def check_gpu():
    result = subprocess.run(
        [
            "python",
            "-c",
            (
                "import torch; "
                "print(torch.cuda.is_available()); "
                "print(torch.cuda.get_device_name(0) "
                "if torch.cuda.is_available() else 'NO_GPU')"
            ),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    lines = result.stdout.strip().splitlines()

    if not lines or lines[0].strip().lower() != "true":
        raise RuntimeError("CUDA GPU is not available.")

    return lines[1] if len(lines) > 1 else "unknown"


def validate_model_files():
    model_dir = Path("/models")

    required = [
        model_dir / "musetalkV15" / "unet.pth",
        model_dir / "musetalkV15" / "musetalk.json",
        model_dir / "sd-vae" / "config.json",
        model_dir / "sd-vae" / "diffusion_pytorch_model.bin",
        model_dir / "whisper" / "config.json",
        model_dir / "whisper" / "pytorch_model.bin",
        model_dir / "dwpose" / "dw-ll_ucoco_384.pth",
        model_dir / "face-parse-bisent" / "79999_iter.pth",
        model_dir / "face-parse-bisent" / "resnet18-5c106cde.pth",
        model_dir / "syncnet" / "latentsync_syncnet.pt",
    ]

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise RuntimeError(
            "MuseTalk model files are missing. "
            "Attach the model storage/cache containing the required weights.",
            missing,
        )

    return True


def run_musetalk(video_path: Path, audio_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = APP_DIR / "configs" / "inference" / "test.yaml"

    command = [
        "python",
        "-m",
        "scripts.inference",
        "--inference_config",
        str(config_path),
        "--result_dir",
        str(output_dir),
        "--unet_model_path",
        "/models/musetalkV15/unet.pth",
        "--unet_config",
        "/models/musetalkV15/musetalk.json",
        "--version",
        "v15",
        "--ffmpeg_path",
        "/usr/bin/ffmpeg",
    ]

    import yaml

    job_config = {
        "task_1to1": {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
        },
        "version": "v15",
        "bbox_shift": 0,
    }

    job_config_path = output_dir / "inference.yaml"

    with job_config_path.open("w") as config_file:
        yaml.safe_dump(job_config, config_file)

    command[5] = str(job_config_path)

    process = subprocess.run(
        command,
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
        timeout=int(os.getenv("JOB_TIMEOUT_SECONDS", "900")),
    )

    if process.returncode != 0:
        raise RuntimeError(
            "MuseTalk inference failed.",
            {
                "stdout": process.stdout[-8000:],
                "stderr": process.stderr[-8000:],
            },
        )

    outputs = sorted(
        [
            p
            for p in output_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mov"}
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not outputs:
        raise RuntimeError(
            "MuseTalk completed but no video output was found.",
            {"stdout": process.stdout[-4000:]},
        )

    return outputs[0]


def handler(job):
    job_input = job.get("input", {})

    job_id = job.get("id") or str(uuid.uuid4())
    work_dir = INPUT_DIR / job_id
    output_dir = OUTPUT_DIR / job_id

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_input = (
            job_input.get("video_url")
            or job_input.get("source_video")
            or job_input.get("image_url")
            or job_input.get("source_image")
        )

        audio_input = (
            job_input.get("audio_url")
            or job_input.get("audio")
            or job_input.get("voice_audio")
        )

        if not video_input:
            return error(
                "A source video or image is required.",
                "MISSING_VIDEO",
            )

        if not audio_input:
            return error(
                "Audio is required.",
                "MISSING_AUDIO",
            )

        gpu_name = check_gpu()

        video_path = get_input_file(
            video_input,
            work_dir / "source.mp4",
        )

        audio_path = get_input_file(
            audio_input,
            work_dir / "speech.wav",
        )

        validate_model_files()

        output_video = run_musetalk(
            video_path,
            audio_path,
            output_dir,
        )

        return {
            "status": "completed",
            "job_id": job_id,
            "gpu": gpu_name,
            "model": "MuseTalk 1.5",
            "output": {
                "filename": output_video.name,
                "path": str(output_video),
            },
        }

    except Exception as exc:
        return error(
            str(exc),
            "INFERENCE_FAILED",
        )


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
        }
    )