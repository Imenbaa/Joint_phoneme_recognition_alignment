# JointRecAlign: step-by-step guide

JointRecAlign listens to French recordings and writes, for each one, a **Praat TextGrid** with:

- the **phones** it hears and their **boundaries** (10 ms resolution),
- the **pauses**,
- for each phone, the **5 most likely phones** according to the model, with their probabilities.

No transcript is needed. This guide needs no programming knowledge: you copy two commands
once, and then use a window or Praat.

## 1. Install (once, about 10 minutes)

You need an internet connection and up to 8 GB of free disk space (less on a Mac).

**On a Mac.** Open the **Terminal** app (Applications › Utilities › Terminal), paste this line,
press Enter, and wait until it finishes:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows.** Open **PowerShell** (Start menu, type "PowerShell"), paste this line, press
Enter, and wait until it finishes:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**On Linux.** Open a terminal and use the Mac line.

This installs `uv`, a small tool that installs JointRecAlign and the Python it needs, without
changing anything else on your computer.

Then **close the Terminal / PowerShell window, open a new one**, and paste:

```
uv tool install --python 3.12 "JointRecAlign @ https://github.com/Imenbaa/Joint_phoneme_recognition_alignment/archive/refs/heads/main.zip"
```

It downloads about 1 to 3 GB of software (several minutes). Finally, download the model
(1.2 GB, also only once):

```
JointRecAlign --download
```

The model is stored in a cache folder (`~/Library/Caches/JointRecAlign` on a Mac,
`%LOCALAPPDATA%\JointRecAlign` on Windows, `~/.cache/JointRecAlign` on Linux). If you skip
this step, it is downloaded the first time you transcribe.

## 2. Transcribe recordings

### With the window

Open a Terminal / PowerShell window and type:

```
JointRecAlign
```

A window opens:

1. **Add files...** or **Add a folder...** to choose your recordings (WAV, FLAC, MP3, OGG or
   AIFF; any sampling rate; mono or stereo).
2. Choose where the results go. By default, a folder `JointRecAlign` is made **next to your
   recordings**, so your own TextGrids are never overwritten.
3. Choose the phone symbols: IPA (`ɑ̃ ɛ̃ ɔ̃ œ`) or the model's one-character codes (`@ 1 § 9`).
4. Click **Run**. The progress appears at the bottom. At the end, click
   **Open the results folder**.

### From Praat

Install the Praat plugin once (in a Terminal / PowerShell window):

```
JointRecAlign --install-praat-plugin
```

Restart Praat. Select one or more **Sound** objects in the Objects window, and click
**JointRecAlign: transcribe...** in the buttons on the right. Praat waits while the model
works (about a few seconds per minute of audio on a recent computer); then a TextGrid with
the same name appears next to each Sound. Select a Sound and its TextGrid and click
**View & Edit**.

### By typing a command

```
JointRecAlign my_recordings/
JointRecAlign recording1.wav recording2.wav
JointRecAlign my_recordings/ -o results/          (results in the folder results/)
```

A folder means every audio file in it. `JointRecAlign --help` lists all the options.

## 3. Read the results

For each recording `name.wav`, the results folder holds:

- `name.TextGrid`, to open in Praat with the recording, with 7 tiers:

  | tier | content |
  |---|---|
  | `phones` | the phone the model hears |
  | `confidence` | the model's probability for that phone |
  | `top2` ... `top5` | the next candidates, e.g. `p 0.050` |
  | `pauses` | `SIL` on every pause |

- `name.csv`, the same as a table (one row per phone), to open in Excel, LibreOffice or R.

Read the probabilities with care: they are not calibrated. The first phone is often given
about 0.99 even when it is wrong; but the **order** of the candidates is informative: when the
first phone is wrong, the right one is often 2nd or 3rd.

The model hears standard French phones. It does not mark diacritics or non-native or
pathological variants.

## 4. Update or remove

```
uv tool upgrade JointRecAlign          (newer version)
uv tool uninstall JointRecAlign        (remove the program)
```

To remove the model too, delete its cache folder (see step 1).

## If something goes wrong

| message | what to do |
|---|---|
| `command not found: JointRecAlign` / `uv` | Close the Terminal / PowerShell window and open a new one. If it persists, run `uv tool update-shell`, then open a new window. |
| `could not download the model` | Check the internet connection and run `JointRecAlign --download` again. |
| `not a supported audio file` | Convert the file to WAV (in Praat: Read from file, then Save as WAV file). |
| `<name>: FAILED (...)` | That recording could not be read; the others are still transcribed. |
| The window does not open (Linux) | `sudo apt install python3-tk`, or use the command. |
| In Praat, no **JointRecAlign** button | Restart Praat after `JointRecAlign --install-praat-plugin`; the button appears when a Sound is selected. |

## Licence

The program is under the MIT licence. The model is under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): free for non-commercial use,
with attribution.
