# zunel/router.py
import os
import zipfile
import requests
from pathlib import Path





_GITHUB_REPO = "rodolfocasan/zunel"
_GITHUB_API_RELEASES = "https://api.github.com/repos/{}/releases".format(_GITHUB_REPO)
_FALLBACK_TAG = "models-1.0.0"

_BASE_DIR = os.path.join(str(Path.home()), ".zunel")
_MODELS_DIR = os.path.join(_BASE_DIR, "models")





def get_storage_path():
    return _BASE_DIR


def _get_latest_models_tag():
    response = requests.get(
        _GITHUB_API_RELEASES,
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "zunel"
        },
        timeout = 30
    )
    response.raise_for_status()
    releases = response.json()

    for release in releases:
        tag = release.get("tag_name", "")
        is_draft = release.get("draft", False)
        is_pre = release.get("prerelease", False)

        if not is_draft and not is_pre and tag.startswith("models-"):
            return tag

    for release in releases:
        tag = release.get("tag_name", "")
        if tag.startswith("models-"):
            return tag
    return _FALLBACK_TAG


def _download_file(url, dest_path):
    print("[zunel] Downloading: {}".format(url))

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        percent = min(int(downloaded * 100 / total), 100)
                        print("\r[zunel] Progress: {}%".format(percent), end="", flush=True)
    print()


def _extract_zip(zip_path, extract_to):
    print("[zunel] Extracting: {}".format(os.path.basename(zip_path)))

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    os.remove(zip_path)
    print("[zunel] Extracted to: {}".format(extract_to))


def _resolve_adapter_path():
    return os.path.join(_MODELS_DIR, "adapters_output", "adapter_best.pth")


def _resolve_model_path():
    return os.path.join(_MODELS_DIR, "zunel_weights", "timbre_engine", "model.pth")


def _resolve_model_config_path():
    return os.path.join(_MODELS_DIR, "zunel_weights", "timbre_engine", "config.json")


def download_models():
    os.makedirs(_MODELS_DIR, exist_ok=True)

    adapter_p = _resolve_adapter_path()
    model_p = _resolve_model_path()
    config_p = _resolve_model_config_path()

    adapters_present = os.path.exists(adapter_p)
    model_present = os.path.exists(model_p) and os.path.exists(config_p)

    if adapters_present and model_present:
        print("[zunel] Models already present at: {}".format(_MODELS_DIR))
        return adapter_p, model_p, config_p

    print("[zunel] Fetching latest models release...")

    try:
        tag = _get_latest_models_tag()
    except Exception as exc:
        print("[zunel] Could not reach GitHub API: {}".format(exc))
        print("[zunel] Falling back to tag: {}".format(_FALLBACK_TAG))
        tag = _FALLBACK_TAG

    print("[zunel] Release tag: {}".format(tag))

    base_url = "https://github.com/{}/releases/download/{}".format(_GITHUB_REPO, tag)

    if not adapters_present:
        adapters_zip = os.path.join(_MODELS_DIR, "adapters.zip")
        adapters_url = "{}/adapters.zip".format(base_url)
        _download_file(adapters_url, adapters_zip)
        _extract_zip(adapters_zip, _MODELS_DIR)
    else:
        print("[zunel] Adapters already present, skipping download.")

    if not model_present:
        weights_zip = os.path.join(_MODELS_DIR, "zunel_weights.zip")
        weights_url = "{}/zunel_weights.zip".format(base_url)
        _download_file(weights_url, weights_zip)
        _extract_zip(weights_zip, _MODELS_DIR)
    else:
        print("[zunel] Model weights already present, skipping download.")

    return adapter_p, model_p, config_p