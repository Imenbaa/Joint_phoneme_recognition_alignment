# JointRecAlign: step-by-step guide

JointRecAlign listens to French recordings and writes, for each one, a **Praat TextGrid** with:

- the **phones** it hears and their **boundaries** (10 ms resolution),
- the **pauses**,
- for each phone, the **5 most likely phones** according to the model, with their probabilities.

No transcript is needed. This guide needs no programming knowledge: you copy two commands
once, then type one command per corpus (or use a button in Praat).

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
JointRecAlign download
```

The model is stored in a cache folder (`~/Library/Caches/JointRecAlign` on a Mac,
`%LOCALAPPDATA%\JointRecAlign` on Windows, `~/.cache/JointRecAlign` on Linux). If you skip
this step, it is downloaded the first time you transcribe.

## 2. Transcribe recordings

Open a Terminal / PowerShell window. Type `JointRecAlign transcribe`, a space, then **drag the
folder of your recordings onto the window** (it writes its path), and press Enter:

```
JointRecAlign transcribe /path/to/my_recordings
```

Every audio file of the folder is transcribed (WAV, FLAC, MP3, OGG or AIFF; any sampling rate;
mono or stereo). The results go to a new folder `JointRecAlign` **inside that folder**, so your
own TextGrids are never overwritten. You can also give single files, or choose the results
folder with `-o`:

```
JointRecAlign transcribe recording1.wav recording2.wav
JointRecAlign transcribe my_recordings -o results
```

**What you see** (an example). First every file is checked, then transcribed, then a summary:

```
Checking 3 recordings
  WARNING my_recordings/ABC_02.wav: sampled at 8000 Hz, below the model's 16 kHz: expect lower accuracy
  ERROR   my_recordings/ABC_03.wav: cannot read the audio (...)
Checked 3 recordings: 2 can be transcribed, 1 cannot, 1 warnings
Loading the model
Running on the CPU
[1/2] ABC_01: 78.4 s of audio, 543 phones, 41 pauses (12.3 s)
[2/2] ABC_02: 65.0 s of audio, 466 phones, 35 pauses (10.1 s)

Summary
  3 recordings: 2 transcribed, 1 NOT transcribed
  not transcribed: my_recordings/ABC_03.wav: cannot read the audio (...)
  warning: my_recordings/ABC_02.wav: sampled at 8000 Hz, below the model's 16 kHz: expect lower accuracy
  results: my_recordings/JointRecAlign
```

A file that cannot be read never stops the others. The same summary is saved in
`JointRecAlign.log` in the results folder.

To only **check** the files, without transcribing (fast, no model):

```
JointRecAlign validate my_recordings
```

### With plots: `--report`

```
JointRecAlign transcribe my_recordings --report
```

also writes one web page per recording, and `index.html` listing them. Double-click
`index.html` (it opens in your browser, no internet needed): a table of the recordings, sortable,
with how many phones each has and how many the model was unsure about. Click a name to see:

- the **waveform** with the recognised **phones** and **pauses** under it; hover over a phone to
  see its 5 candidates, click it to hear it;
- the **attention** of the model: which phone each moment of the audio is attached to; the
  line is the path that gives the boundaries;
- the **phone posteriors**: how probable every phone is at each moment;
- the **top 5** of each moment.

Drag to scroll, ⌘/Ctrl + mouse wheel to zoom, space to play. The address of the page follows the
view (e.g. `ABC_01.html#12.5-14.5`): copy it to show someone the same place. The pages play the
audio from where it was when the report was made: keep the recordings in place.

### From Praat

Install the Praat plugin once (in a Terminal / PowerShell window):

```
JointRecAlign install-praat-plugin
```

Restart Praat. Select one or more **Sound** objects in the Objects window, and click
**JointRecAlign: transcribe...** in the buttons on the right. Praat waits while the model
works (a few seconds per minute of audio on a recent computer); then a TextGrid with the same
name appears next to each Sound. Select a Sound and its TextGrid and click **View & Edit**.
This suits one recording at a time; for a whole corpus, use the command.

`JointRecAlign --help` and `JointRecAlign transcribe --help` list all the options.

## 3. Read the results

For each recording `name.wav`, the results folder holds:

- `name.TextGrid`, to open in Praat with the recording, with 7 tiers:

  | tier | content |
  |---|---|
  | `phones` | the phone the model hears |
  | `confidence` | the model's probability for that phone |
  | `top2` ... `top5` | the next candidates, e.g. `p 0.050` |
  | `pauses` | `SIL` on every pause |

- `name.csv`, the same as a table (one row per phone), to open in Excel, LibreOffice or R;
- with `--report`: `name.html` and `index.html` (see above);
- `JointRecAlign.log`: the summary of the run.

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
| `could not download the model` | Check the internet connection and run `JointRecAlign download` again. |
| `not a supported audio file` | Convert the file to WAV (in Praat: Read from file, then Save as WAV file). |
| `<name>: FAILED (...)` | That recording could not be read; the others are still transcribed. |
| `<file>: FAILED (...)` or `ERROR ... cannot read the audio` | That recording could not be read; convert it to WAV (in Praat: Read from file, then Save as WAV file). The others are transcribed. |
| The report plays no sound | The recording was moved or renamed after the report was made; run the command again. |
| In Praat, no **JointRecAlign** button | Restart Praat after `JointRecAlign install-praat-plugin`; the button appears when a Sound is selected. |

## Licence

The program is under the MIT licence. The model is under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): free for non-commercial use,
with attribution.
