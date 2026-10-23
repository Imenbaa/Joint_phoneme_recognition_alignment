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

**CTC posterior**:
The CTC head's probability of each phone (and of the blank) at one frame; sharp spikes on one or two frames per phone.
_Avoid_: CTC score, emission

## Alignment

**Alignment**:
The assignment of every frame to exactly one phone, in order, so that each phone covers a contiguous run of frames.
_Avoid_: segmentation, forced alignment (unless the CTC forced aligner is meant)

**Target phone sequence**:
The phones the aligner places in time: the reference transcript during training, the model's own CTC decode at inference.
_Avoid_: text, transcript, labels

**Phone key**:
One vector per phone of the target sequence, made from the encoder frames that phone most likely covers.
_Avoid_: phone embedding, text encoding

**Phone centre**:
The frame where a target phone is believed to sit before alignment, used to localise its key.

**Query**:
One vector per frame describing the local acoustics, compared with every phone key.
_Avoid_: frame embedding

**Attention energy**:
The frame × phone score matrix comparing every query with every phone key; the raw material of the alignment.
_Avoid_: attention logits, similarity matrix

**Alignment prior**:
A frame × phone preference for the diagonal (early frames with early phones), added to the attention energy and faded out during training.
_Avoid_: diagonal prior, guided attention

## Training stages

**Stage**:
One of the three successive training runs (CTC 20 ms, CTC 10 ms, joint aligner), each starting from the previous one.
_Avoid_: phase, step
