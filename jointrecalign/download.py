"""The released model: downloaded once, on first use, into a cache folder.

    Linux    ~/.cache/JointRecAlign
    macOS    ~/Library/Caches/JointRecAlign
    Windows  %LOCALAPPDATA%\\JointRecAlign

JOINTRECALIGN_HOME overrides the folder; JOINTRECALIGN_MODEL_URL the download URL.
"""
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile

MODEL_NAME = "aligner_10ms_sil_alpha0.1"
MODEL_URL = ("https://github.com/Imenbaa/Joint_phoneme_recognition_alignment/releases/download/"
             f"v0.1.0/{MODEL_NAME}.zip")
MODEL_SHA256 = "a1afc8f77544475e062b9de252abfe2d1b9f10c79f5b9f4e702c57548ee67f7a"


def cache_dir():
    if os.environ.get("JOINTRECALIGN_HOME"):
        return os.environ["JOINTRECALIGN_HOME"]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(base, "JointRecAlign")


def _fetch(url, dest, log):
    """Download url to dest, printing the progress every 5 %."""
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown = 0, -5
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total and 100 * done // total >= shown + 5:
                shown = 100 * done // total
                log(f"  {shown:3d} %  ({done / 1e6:.0f} / {total / 1e6:.0f} MB)")


def default_model(log=print):
    """Folder of the released model; downloaded, checked and unzipped the first time."""
    root = cache_dir()
    folder = os.path.join(root, MODEL_NAME)
    if os.path.exists(os.path.join(folder, "model.safetensors")):
        return folder
    os.makedirs(root, exist_ok=True)
    url = os.environ.get("JOINTRECALIGN_MODEL_URL", MODEL_URL)
    part = os.path.join(root, MODEL_NAME + ".zip.part")
    log(f"First use: downloading the model (1.2 GB, only once) into {root}")
    try:
        _fetch(url, part, log)
    except Exception as e:
        raise RuntimeError(f"could not download the model from {url} ({e}).\n"
                           "Check the internet connection and try again, or download the zip by hand, "
                           "unzip it and give its folder with --model.") from None
    h = hashlib.sha256()
    with open(part, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != MODEL_SHA256 and "JOINTRECALIGN_MODEL_URL" not in os.environ:
        os.remove(part)
        raise RuntimeError("the downloaded model is damaged (checksum mismatch); please try again.")
    log("Unzipping the model")
    tmp = folder + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(part) as z:
        z.extractall(tmp)
    shutil.rmtree(folder, ignore_errors=True)
    os.replace(os.path.join(tmp, MODEL_NAME), folder)
    shutil.rmtree(tmp, ignore_errors=True)
    os.remove(part)
    return folder
