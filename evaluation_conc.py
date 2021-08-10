import os
from os import listdir
import torchaudio
import torch
import re
import librosa
from datasets import Dataset, load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings
from os.path import join
import numpy as np
LANG_ID = "it"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
DEVICE = "cuda"

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]

test_dataset = load_dataset("common_voice", LANG_ID, split="test")
size = len(test_dataset)
wer = load_metric("wer")
cer = load_metric("cer")
chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.to(DEVICE)
# Preprocessing the datasets.
# We need to read the audio files as arrays
def evaluate(speech):
    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    pred_strings = processor.batch_decode(pred_ids)
    return pred_strings





silence, sampling_rate = torchaudio.load("silence.mp3")
resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
silence = resampler.forward(silence.squeeze(0)).numpy()

print("CONCATENATE",flush=True)
i = 0
total_wer = 0
total_cer = 0
while i < len(test_dataset["path"]):
    print(i,flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech, sampling_rate = torchaudio.load(test_dataset["path"][i] )

        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        speech_array = resampler.forward(speech.squeeze(0)).numpy()
    sentence = re.sub(chars_to_ignore_regex, "", test_dataset["sentence"][i]).upper()
    
    first_audio = speech_array.tolist()
    first_sent = sentence
    second_audio = None
    j = i+1
    while j < len(test_dataset["path"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            speech, sampling_rate = torchaudio.load(test_dataset["path"][i] )

            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
            speech_array = resampler.forward(speech.squeeze(0)).numpy()
        sentence = re.sub(chars_to_ignore_regex, "", test_dataset["sentence"][i]).upper()
        second_audio = speech_array.tolist()
        second_sent = sentence

        conc_audio = first_audio + silence.tolist() 
        conc_audio = conc_audio + second_audio
        conc_sent = first_sent + " " + second_sent
        prediction = evaluate(conc_audio)
        total_wer += wer.compute(predictions=[prediction[0].upper()], references=[conc_sent.upper()]) * 100
        total_cer += cer.compute(predictions=[prediction[0].upper()], references=[conc_sent.upper()]) * 100
        j +=1
    i += 1
    if second_audio == None:
        prediction = evaluate(first_audio)
        total_wer += wer.compute(predictions=[prediction[0].upper()], references=[first_sent.upper()]) * 100
        total_cer += cer.compute(predictions=[prediction[0].upper()], references=[first_sent.upper()]) * 100


size = (size*size + size)/2
total_wer /= size
total_cer /= size



print(f"WER: {total_wer}")
print(f"CER: {total_cer}")


exit(1)




###############################







def evaluate_mlls(speech):
    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)


wer_result=wer.compute(predictions=predictions, references=references) * 100
cer_result=cer.compute(predictions=predictions, references=references) * 100
with open("wer_commonvoice.txt", "w") as f:
    f.write(f"WER: {wer_result}\nCER: {cer_result}")



main_dir = "dataset/MultilingualLibriSpeech/mls_italian_opus"

test_dir = join(main_dir, 'test')


print("MLLS-ita")

total_cer = 0
total_wer = 0
labels_dict = {}
audio_dict = {}
listOfOpus = list()

for (dirpath, dirnames, filenames) in os.walk(test_dir):
    listOfOpus += [os.path.join(dirpath, file) for file in filenames if file.endswith(".opus")]

listOfTxt = list()

for (dirpath, dirnames, filenames) in os.walk(test_dir):
    listOfTxt += [os.path.join(dirpath, file) for file in filenames if file == "transcripts.txt"]

for transcript in listOfTxt:
    with open(transcript, 'r') as f:#
        content = f.read()
        sentences = content.split(sep="\n")


    for sent in sentences:
        if(sent != ''):
            sent = re.sub(' +', ' ', sent)
            sent = sent.split("\t", maxsplit=1)
            labels_dict[sent[0]] = sent[1]

for opus in listOfOpus:
    audio_dict[opus.split("/")[-1].split(".")[0]] = opus

dict_dataset = {'path': [], 'sentence': []}

for k, v in audio_dict.items():
    dict_dataset['path'].append(v)
    dict_dataset['sentence'].append(labels_dict[k])

hug_dataset = Dataset.from_dict(dict_dataset)
for batch in hug_dataset:
    speech, sampling_rate = torchaudio.load(batch["path"])
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
    speech_array = resampler.forward(speech.squeeze(0)).numpy()
    batch["speech"] = speech_array
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()

    result = evaluate_mlls(batch["speech"])
    predictions = [result[0].upper()]
    references = [batch["sentence"].upper()]

    total_wer += wer.compute(predictions=predictions, references=references) * 100
    total_cer += cer.compute(predictions=predictions, references=references) * 100

with open("wer_mlls.txt", "w") as f:
    f.write(f"WER: {total_wer/len(hug_dataset)}\nCER: {total_cer/len(hug_dataset)}")
