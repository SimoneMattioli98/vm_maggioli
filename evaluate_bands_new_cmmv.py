import re
from os.path import isfile, join
from os import listdir, walk
from datasets import Dataset, load_dataset, load_metric, concatenate_datasets
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
import torch 
import random
import pandas as pd
import csv
import warnings

def show_random_elements(dataset, num_examples=1):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    print(df)



print("####CMMV")

def create_new_cmmv(path_to_csv, audio_folder):
    dict = {"path": [], "sentence": []}
    with open(path_to_csv, 'r') as f:

        read_tsv = csv.reader(f, delimiter="\t")
        read_tsv = [row for row in read_tsv]
        read_tsv = read_tsv[1:]
        for row in read_tsv:
            dict["path"].append(f"{audio_folder}{row[1]}")
            dict["sentence"].append(row[2])
    return Dataset.from_dict(dict)

print("#### Loading dataset")

print("####Commonvoice new")

fold = "dataset/cv_new/cv-corpus-7.0-2021-07-21/it/"


common_voice_train = create_new_cmmv(f"{fold}train2.csv", f"{fold}clips2/")
common_voice_val = create_new_cmmv(f"{fold}dev2.csv", f"{fold}clips2/")   

print("####Merge datasets")

merged_dataset = concatenate_datasets([common_voice_train, common_voice_val]).shuffle(seed=1234)



CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", ";", ":", '""', "%", '"', "?", "?", "·", "?", "~", "?",
                   "?", "?", "?", "?", "«", "»", "„", "“", "”", "?", "?", "‘", "’", "«", "»", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "?", "‹", "›", "©", "®", "—", "?", "?",
                   "?", "?", "?", "?", "~", "?", ",", "{", "}", "(", ")", "[", "]", "?", "?", "?", "?",
                   "?", "?", "?", "?", "?", "?", "?", ":", "!", "?", "?", "?", "/", "\\", "º", "-", "^", "?", "ˆ"]

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"


def remove_special_characters_comm(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).strip().upper() + " "
    return batch

merged_dataset = merged_dataset.map(remove_special_characters_comm)

show_random_elements(merged_dataset, 4)

DEVICE = "cuda"

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-italian")

model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-italian").to(DEVICE)

total_wer = 0
#total_cer = 0

wer = load_metric("wer")
cer = load_metric("cer")

ranges = {} # contains (count, tot_wer)

bands_len = 2 #2 second bands


for index, batch in enumerate(merged_dataset):
    print(index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()

    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    prediction = processor.batch_decode(pred_ids)

    #print(f"PRED: {prediction[0].upper()}\nREF: {batch['sentence'].upper()}")
    
    wer_computed = wer.compute(predictions=[prediction[0].upper()], references=[batch["sentence"].upper()]) * 100
    total_wer += wer_computed
    #total_cer += cer.compute(predictions=[prediction[0].upper()], references=[batch["sentence"].upper()]) * 100

    info = torchaudio.info(batch["path"])
    duration_sec = info.num_frames / info.sample_rate

    band = int(duration_sec / bands_len)

    if band not in ranges:
        ranges[band] = [1, wer_computed]
    else:
        ranges[band][0] += 1
        ranges[band][1] += wer_computed

#total_cer /= len(common_voice_test)
total_wer /= len(merged_dataset)

with open(f"evaluation_bands_new_cmmv.txt", "w") as f:
    for key in sorted(ranges.keys()):
        mean_wer = ranges[key][1] / ranges[key][0]
        f.write(f"[{int(key)*bands_len},{int(key)*bands_len+bands_len}) -> Count: {ranges[key][0]}, Wer: {mean_wer}\n")
    f.write(f"WER: {total_wer}\n") #CER: {total_cer}")




