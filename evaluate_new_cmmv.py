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

def create_new_cmmv(path_to_csv):
    dict = {"path": [], "sentence": []}
    with open(path_to_csv, 'r') as f:

        read_tsv = csv.reader(f, delimiter="\t")
        read_tsv = [row for row in read_tsv]
        read_tsv = read_tsv[1:]
        for row in read_tsv:
            dict["path"].append(row[1])
            dict["sentence"].append(row[2])
    return Dataset.from_dict(dict)

print("#### Loading dataset")

print("####Commonvoice new")

fold = "../dataset/cv_new/cv-corpus-7.0-2021-07-21/it/"



common_voice_test = create_new_cmmv(f"{fold}test2.csv")


CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", ";", ":", '""', "%", '"', "?", "?", "·", "?", "~", "?",
                   "?", "?", "?", "?", "«", "»", "„", "“", "”", "?", "?", "‘", "’", "«", "»", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "?", "‹", "›", "©", "®", "—", "?", "?",
                   "?", "?", "?", "?", "~", "?", ",", "{", "}", "(", ")", "[", "]", "?", "?", "?", "?",
                   "?", "?", "?", "?", "?", "?", "?", ":", "!", "?", "?", "?", "/", "\\", "º", "-", "^", "?", "ˆ"]

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"


def remove_special_characters_comm(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).strip().upper() + " "
    return batch

common_voice_test = common_voice_test.map(remove_special_characters_comm)

show_random_elements(common_voice_test, 4)

DEVICE = "cuda"

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-italian")

model = Wav2Vec2ForCTC.from_pretrained("../finetuning_jonatas_new_cmmv/final").to(DEVICE)

total_wer = 0
total_cer = 0

wer = load_metric("wer")
cer = load_metric("cer")

for index, batch in enumerate(common_voice_test):
    print(index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sampling_rate = librosa.load(f"{fold}/clips2/"+batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()

    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    prediction = processor.batch_decode(pred_ids)

    #print(f"PRED: {prediction[0].upper()}\nREF: {batch['sentence'].upper()}")

    total_wer += wer.compute(predictions=[prediction[0].upper()], references=[batch["sentence"].upper()]) * 100
    total_cer += cer.compute(predictions=[prediction[0].upper()], references=[batch["sentence"].upper()]) * 100

total_cer /= len(common_voice_test)
total_wer /= len(common_voice_test)

with open(f"results_new_cmmv.txt", "w") as f:
    f.write(f"WER: {total_wer}\nCER: {total_cer}")




