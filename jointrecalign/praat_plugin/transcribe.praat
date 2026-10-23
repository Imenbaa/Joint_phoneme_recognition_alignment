# JointRecAlign: phones, boundaries, pauses and top-5 candidates of the selected Sounds,
# as TextGrids added to the object list. No transcript is needed.
# The line "command$ = ..." is written by `JointRecAlign --install-praat-plugin`.
command$ = "@COMMAND@"

form JointRecAlign: transcribe
    comment Phones, boundaries, pauses and the 5 best candidates of each phone.
    comment No transcript is needed. Praat waits until the model has finished.
    optionmenu Phone_symbols 1
        option IPA (ɑ̃ ɛ̃ ɔ̃ œ)
        option model codes (@ 1 § 9)
endform
symbols$ = if phone_symbols = 1 then "ipa" else "model" fi

n = numberOfSelected ("Sound")
for i to n
    sound[i] = selected ("Sound", i)
endfor
folder$ = temporaryDirectory$ + "/JointRecAlign_praat"
createFolder: folder$
files$ = ""
for i to n
    selectObject: sound[i]
    name$[i] = selected$ ("Sound")
    file$[i] = folder$ + "/sound" + string$ (i)
    deleteFile: file$[i] + ".TextGrid"
    Save as WAV file: file$[i] + ".wav"
    files$ = files$ + " """ + file$[i] + ".wav"""
endfor

runSystem: command$, " --symbols ", symbols$, " -o """, folder$, """", files$

for i to n
    textgrid[i] = Read from file: file$[i] + ".TextGrid"
    Rename: name$[i]
    deleteFile: file$[i] + ".wav"
    deleteFile: file$[i] + ".TextGrid"
    deleteFile: file$[i] + ".csv"
endfor
selectObject: sound[1], textgrid[1]
for i from 2 to n
    plusObject: sound[i], textgrid[i]
endfor
