"""Install the Praat plugin: a "JointRecAlign: transcribe..." button for selected Sounds.

    JointRecAlign install-praat-plugin

The plugin folder plugin_JointRecAlign goes into Praat's preferences folder
(PRAAT_PREFS_DIR overrides it); Praat reads it when it starts.
"""
import os
import shutil
import sys

PLUGIN = "plugin_JointRecAlign"


def praat_prefs_dir():
    if os.environ.get("PRAAT_PREFS_DIR"):
        return os.environ["PRAAT_PREFS_DIR"]
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        return os.path.join(home, "Praat")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Preferences", "Praat Prefs")
    return os.path.join(home, ".praat-dir")


def _quoted(path):
    return f'""{path}""' if " " in path else path        # "" is a quote inside a Praat string


def install_plugin():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "praat_plugin")
    dest = os.path.join(praat_prefs_dir(), PLUGIN)
    os.makedirs(dest, exist_ok=True)
    shutil.copy(os.path.join(src, "setup.praat"), dest)
    command = f"{_quoted(sys.executable)} -m jointrecalign"
    script = open(os.path.join(src, "transcribe.praat"), encoding="utf-8").read()
    with open(os.path.join(dest, "transcribe.praat"), "w", encoding="utf-8") as f:
        f.write(script.replace("@COMMAND@", command))
    print(f"Praat plugin installed in {dest}\n"
          "Restart Praat, select one or more Sounds, and click \"JointRecAlign: transcribe...\".\n"
          "The first time, the model is downloaded (1.2 GB); to do that now: JointRecAlign download")
