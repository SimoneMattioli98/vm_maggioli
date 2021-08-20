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


print("#### Loading dataset")

print("####Commonvoice")
common_voice_test = load_dataset("common_voice", "it", split=f'test')

common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

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

#show_random_elements(hug_dataset_test, 2)

common_voice_test = common_voice_test.map(remove_special_characters_comm)
# print(common_voice_train["sentence"][0])
#print(hug_dataset_test)
# print(hug_dataset_val)
print(common_voice_test)
# print(common_voice_val)

show_random_elements(common_voice_test, 2)


print("####Merge datasets")

merged_dataset_test = concatenate_datasets([hug_dataset_test, common_voice_test]).shuffle(seed=1234)

show_random_elements(merged_dataset_test, 2)

DEVICE = "cuda"

model_name = "jonatasgrosman"

processor = Wav2Vec2Processor.from_pretrained(f"{model_name}/wav2vec2-large-xlsr-53-italian")

model = Wav2Vec2ForCTC.from_pretrained(f"{model_name}/wav2vec2-large-xlsr-53-italian").to(DEVICE)

wer = load_metric("wer")

ranges = {} # contains (count, tot_wer)

bands_len = 3 #2 second bands

print("#### EVALUATE")
for index, batch in enumerate(merged_dataset_test):
    print(f"{index} - {len(merged_dataset_test)}")
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
    wer_computed = wer.compute(predictions=[prediction[0].upper()], references=[batch["sentence"].upper()]) * 100

    info = torchaudio.info(batch["path"])
    duration_sec = info.num_frames / sampling_rate

    band = int(duration_sec / bands_len)

    if band not in ranges:
        ranges[band] = [1, wer_computed]
    else:
        ranges[band][0] += 1
        ranges[band][1] += wer_computed

with open(f"evaluation_bands_{model_name}.txt", "w") as f:
    for key in sorted(ranges.keys()):
        mean_wer = ranges[key][1] / ranges[key][0]
        f.write(f"[{int(key)*2},{int(key)*2+band}) -> Count: {ranges[key][0]}, Wer: {mean_wer}\n")




    

    