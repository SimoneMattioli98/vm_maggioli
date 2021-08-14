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



print("####MLLS")

main_dir = "dataset/MultilingualLibriSpeech/mls_italian_opus"

test_dir = join(main_dir, 'test')

file_transcripts = 'transcripts.txt'

list_txt_train = []
list_txt_val = []

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", ";", ":", '""', "%", '"', "?", "?", "·", "?", "~", "?",
                   "?", "?", "?", "?", "«", "»", "„", "“", "”", "?", "?", "‘", "’", "«", "»", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "?", "‹", "›", "©", "®", "—", "?", "?",
                   "?", "?", "?", "?", "~", "?", ",", "{", "}", "(", ")", "[", "]", "?", "?", "?", "?",
                   "?", "?", "?", "?", "?", "?", "?", ":", "!", "?", "?", "?", "/", "\\", "º", "-", "^", "?", "ˆ"]

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"

def remove_special_characters_mlls(sentence):
    sentence = re.sub(chars_to_ignore_regex, "", sentence).strip().upper() + " "
    return sentence

def remove_special_characters_comm(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).strip().upper() + " "
    return batch

def create_hug_dataset(split_directory):
    
    list_opus = []
    labels_dict = {}
    for (dirpath, dirnames, filenames) in walk(split_directory):
        list_opus += [join(dirpath, file) for file in filenames if file.endswith(".opus")]

    with open(join(split_directory, file_transcripts), 'r') as f: 
        content = f.read()
        sentences = content.split(sep="\n")

    for sent in sentences:
        if(sent != ''):
            sent = re.sub(' +', ' ', sent)
            sent = sent.split("\t", maxsplit=1)
            labels_dict[sent[0]] = sent[1]

    audio_dict = {opus.split("/")[-1].split(".")[0]: opus for opus in list_opus}

    print("#### Removing special characters from labels mlls")

    labels_dict = {k: remove_special_characters_mlls(v) for k, v in labels_dict.items()}
    dict_dataset = {'path': [], 'sentence': []}

    for k, v in audio_dict.items():
        dict_dataset['path'].append(v)
        dict_dataset['sentence'].append(labels_dict[k])

    tot_len = len(dict_dataset["path"])
    print(f"N DATA TEST: {tot_len}")

    return Dataset.from_dict(dict_dataset)


hug_dataset_test = create_hug_dataset(test_dir)

DEVICE = "cuda"

model_folder = "./finetuning_fbxlsr53_mlls/"

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-italian")

model = Wav2Vec2ForCTC.from_pretrained(f"{model_folder}final").to(DEVICE)

show_random_elements(hug_dataset_test, 4)

total_wer = 0
total_cer = 0

wer = load_metric("wer")
cer = load_metric("cer")

for batch in hug_dataset_test:
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

    total_wer += wer.compute(predictions=[prediction[0].upper()], references=[batch["sentence"].upper()]) * 100
    total_cer += cer.compute(predictions=[prediction[0].upper()], references=[batch["sentence"].upper()]) * 100

total_cer /= len(hug_dataset_test)
total_wer /= len(hug_dataset_test)

with open(f"{model_folder}results.txt", "w") as f:
    f.write(f"WER: {total_wer}\nCER: {total_cer}")




