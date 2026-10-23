# JointRecAlign

Joint phone recognition and phone alignment for French speech: one wav2vec2 model that both
transcribes speech into phones and places each phone in time.

## Signal and units

**Frame**:
One time step of the encoder output; 20 ms in stage 1, 10 ms from stage 2 on.
_Avoid_: step, hop, timestep

**Phone**:
One symbol of the phone vocabulary, the unit the model recognises and aligns.
_Avoid_: phoneme, token, label

**Phone vocabulary**:
The fixed set of phone symbols the model can output, plus the special pad/blank and unknown symbols.
_Avoid_: tokenizer, dictionary

**Utterance**:
One training or evaluation recording with its phone sequence.
_Avoid_: sample, example, clip

## Recognition

**CTC**:
Connectionist Temporal Classification: the training criterion that learns a phone sequence from audio without knowing where each phone is.

**Blank**:
The CTC "no new phone here" symbol; in this project it is the same symbol as padding (`[PAD]`).
_Avoid_: epsilon

**CTC collapse**:
Turning a frame-by-frame CTC output into a phone sequence by merging repeats and dropping blanks.
_Avoid_: CTC decoding, greedy decode (when the collapse step itself is meant)

**PER**:
Phone Error Rate: edit distance between the recognised and the reference phone sequences, divided by the reference length.

## Training stages

**Stage**:
One of the three successive training runs (CTC 20 ms, CTC 10 ms, joint aligner), each starting from the previous one.
_Avoid_: phase, step
