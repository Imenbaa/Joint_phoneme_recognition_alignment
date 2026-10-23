"""A small window for JointRecAlign: choose recordings, click Run.

Opened by `JointRecAlign` with no argument (or `python -m jointrecalign`).
"""
import os
import queue
import subprocess
import sys
import threading

from .transcribe_cli import AUDIO, OUT_NAME, find_audio, out_dir_for, run


def open_folder(path):
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.root = tk, root
        self.files, self.messages, self.busy, self.last_out = [], queue.Queue(), False, None
        root.title("JointRecAlign: phone recognition and alignment")
        root.minsize(640, 520)
        f = ttk.Frame(root, padding=12)
        f.pack(fill="both", expand=True)

        ttk.Label(f, text="1. Recordings (WAV, FLAC, MP3, OGG, AIFF)").pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="Add files...", command=self.add_files).pack(side="left")
        ttk.Button(row, text="Add a folder...", command=self.add_folder).pack(side="left", padx=6)
        ttk.Button(row, text="Clear", command=self.clear).pack(side="left")
        self.listbox = tk.Listbox(f, height=7)
        self.listbox.pack(fill="both", expand=True)

        ttk.Label(f, text="2. Where to save the TextGrids").pack(anchor="w", pady=(10, 0))
        self.out_mode = tk.StringVar(value="next")
        self.out_dir = tk.StringVar(value="")
        ttk.Radiobutton(f, text=f"in a folder {OUT_NAME} next to the recordings", value="next",
                        variable=self.out_mode).pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Radiobutton(row, text="in this folder:", value="choose", variable=self.out_mode).pack(side="left")
        ttk.Entry(row, textvariable=self.out_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Choose...", command=self.choose_out).pack(side="left")

        ttk.Label(f, text="3. Phone symbols").pack(anchor="w", pady=(10, 0))
        self.symbols = tk.StringVar(value="ipa")
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Radiobutton(row, text="IPA (ɑ̃ ɛ̃ ɔ̃ œ)", value="ipa", variable=self.symbols).pack(side="left")
        ttk.Radiobutton(row, text="model codes (@ 1 § 9)", value="model",
                        variable=self.symbols).pack(side="left", padx=12)

        row = ttk.Frame(f)
        row.pack(fill="x", pady=10)
        self.run_button = ttk.Button(row, text="Run", command=self.start)
        self.run_button.pack(side="left")
        self.open_button = ttk.Button(row, text="Open the results folder", state="disabled",
                                      command=lambda: open_folder(self.last_out))
        self.open_button.pack(side="left", padx=6)

        self.log_box = tk.Text(f, height=10, state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True)
        self.root.after(100, self.poll)

    def add(self, paths):
        files, problems = find_audio(paths)
        for p in files:
            p = os.path.abspath(p)
            if p not in self.files:
                self.files.append(p)
                self.listbox.insert("end", p)
        for p in problems:
            self.log(p)

    def add_files(self):
        from tkinter import filedialog
        pattern = " ".join("*" + e for e in AUDIO)
        self.add(filedialog.askopenfilenames(title="Recordings",
                                             filetypes=[("Audio", pattern), ("All files", "*")]))

    def add_folder(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Folder of recordings")
        if d:
            self.add([d])

    def clear(self):
        if not self.busy:
            self.files = []
            self.listbox.delete(0, "end")

    def choose_out(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Folder for the results")
        if d:
            self.out_dir.set(d)
            self.out_mode.set("choose")

    def log(self, text):
        self.messages.put(text)

    def poll(self):
        while not self.messages.empty():
            text = self.messages.get()
            if text is None:                       # end of a run
                self.busy = False
                self.run_button.configure(state="normal")
                if self.last_out and os.path.isdir(self.last_out):
                    self.open_button.configure(state="normal")
                continue
            self.log_box.configure(state="normal")
            self.log_box.insert("end", text + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(100, self.poll)

    def start(self):
        from tkinter import messagebox
        if self.busy:
            return
        if not self.files:
            messagebox.showinfo("JointRecAlign", "Add recordings first.")
            return
        out = None
        if self.out_mode.get() == "choose":
            out = self.out_dir.get().strip()
            if not out:
                messagebox.showinfo("JointRecAlign", "Choose the folder for the results.")
                return
        self.busy = True
        self.run_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.last_out = out_dir_for(self.files[0], out)
        threading.Thread(target=self.work, args=(list(self.files), out, self.symbols.get()), daemon=True).start()

    def work(self, files, out, symbols):
        try:
            run(files, out, symbols=symbols, log=self.log)
        except Exception as e:
            self.log(f"ERROR: {e}")
        self.messages.put(None)


def main():
    try:
        import tkinter as tk
    except ImportError:
        sys.exit("The window needs tkinter, which this Python does not have "
                 "(on Ubuntu/Debian: sudo apt install python3-tk).\n"
                 "The command line works without it: JointRecAlign --help")
    try:
        root = tk.Tk()
    except tk.TclError as e:
        sys.exit(f"Cannot open a window here ({e}). Use the command line: JointRecAlign --help")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
