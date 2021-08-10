from audiomentations import (
    Compose,
    AddGaussianNoise,
    AddGaussianSNR,
    ClippingDistortion,
    FrequencyMask,
    Gain,
    LoudnessNormalization,
    Normalize,
    PitchShift,
    PolarityInversion,
    Shift,
    TimeMask,
    TimeStretch,
)
import time
import torchaudio
import json
import re
from transformers import Trainer
from transformers import TrainingArguments
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

from datasets import Dataset, load_dataset, load_metric, concatenate_datasets
import argparse
from os import listdir, walk
from os.path import isfile, join
import IPython.display as ipd
import numpy as np
import random
import librosa
import numpy as np
import reprlib

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2ForCTC
 

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __init__(self, processor, padding=True, apply_gaussian_noise_with_p=0.5, apply_gain_with_p=0.5, apply_pitch_shift_with_p=0.5,
                 apply_time_stretch_with_p=0.5, sample_rate=16_000):
        self.processor = processor
        self.padding = padding
        self.apply_gaussian_noise_with_p = apply_gaussian_noise_with_p
        self.apply_gain_with_p = apply_gain_with_p
        self.apply_pitch_shift_with_p = apply_pitch_shift_with_p
        self.apply_time_stretch_with_p = apply_time_stretch_with_p
        self.sample_rate = sample_rate

        self.augmentator = None
        if self.apply_gaussian_noise_with_p + self.apply_gain_with_p + self.apply_pitch_shift_with_p + self.apply_time_stretch_with_p > 0:
            self.augmentator = Compose([
                TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=self.apply_time_stretch_with_p),
                PitchShift(min_semitones=-1, max_semitones=1, p=self.apply_pitch_shift_with_p),
                Gain(min_gain_in_db=-1, max_gain_in_db=1, p=self.apply_gain_with_p),
                AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.001, p=self.apply_gaussian_noise_with_p),
            ])

    def _apply_augmentation(self, input_values: List[float]):
        """apply some audio augmentations in the given input_values"""
        if self.augmentator is not None:
            return self.augmentator(samples=np.array(input_values), sample_rate=self.sample_rate).tolist()
        else:
            return input_values
   def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        input_features = [{"input_values": self._apply_augmentation(feature["input_values"])} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

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






def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}





def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}





common_voice_train = load_dataset("common_voice", "it", split=f'train+validation')
common_voice_val = load_dataset("common_voice", "it", split=f'test')

common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_val = common_voice_val.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

show_random_elements(common_voice_train)


CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"


def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
    return batch

print("#### Removing special characters from labels")
common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_val = common_voice_val.map(remove_special_characters)


show_random_elements(common_voice_train)


#print("#### Creating vocabolary")
#vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
#vocab_val = common_voice_val.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_val.column_names)

#print(vocab_train)

#vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_val["vocab"][0]))

#vocab_dict = {v: k for k, v in enumerate(vocab_list)}
#print(vocab_dict)



#vocab_dict["|"] = vocab_dict[" "]
#del vocab_dict[" "]
#vocab_dict["[UNK]"] = len(vocab_dict)
#vocab_dict["[PAD]"] = len(vocab_dict)
#print(len(vocab_dict))
#print(vocab_dict)


#with open('./finetuning_fb_ita_commonvoice/vocab.json', 'r') as vocab_file:
#     print(vocab_file.read())
#    json.dump(vocab_dict, vocab_file)


#print("#### Creating Tokenizer, Feature extractor and Processor")
#tokenizer = Wav2Vec2CTCTokenizer("./finetuning_fb_ita_commonvoice/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")


#feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)


#processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


#processor.save_pretrained('./finetuning_fb_ita_commonvoice/processor')

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-italian")

data_collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding=True,
    apply_gaussian_noise_with_p=0.5,
    apply_gain_with_p=0.5,
    apply_pitch_shift_with_p=0.5,
    apply_time_stretch_with_p=0.5,
    sample_rate=16_000,
)

wer_metric = load_metric("wer")


print("#### Creating model")
model = Wav2Vec2ForCTC.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    attention_dropout=0.1,
    activation_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ctc_zero_infinity=True
)

model.freeze_feature_extractor()


from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.preprocessing(item)


    def preprocessing(self, item):
        speech_array, sampling_rate = torchaudio.load(item["path"])
        item["sampling_rate"] = sampling_rate
        item["target_text"] = item["sentence"]
        del item["path"], item["sentence"]
        resampler = torchaudio.transforms.Resample(orig_freq=item["sampling_rate"], new_freq=16_000)
        speech_array = resampler.forward(speech_array.squeeze(0)).numpy()
        item["speech"] = speech_array
        item["sampling_rate"] = 16_000
        item["input_values"] = self.processor(item["speech"], sampling_rate=item["sampling_rate"]).input_values[0]

        with self.processor.as_target_processor():
           item["labels"] = self.processor(item["target_text"]).input_ids
        del item["target_text"], item["speech"], item["sampling_rate"]
        return item

class CustomDatasetConcatenate(Dataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.list_dataset = list(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item1 = self.dataset[idx]
        try:

            item2 = self.list_dataset[self.list_dataset(idx) + 1]
        except (ValueError, IndexError):
            item2 = None
        return self.preprocessing(item1, item2)

    def preprocessing(self, item1, item2):
        speech_array, sampling_rate = torchaudio.load(item1["path"])
        item1["sampling_rate"] = sampling_rate
        item1["target_text"] = item1["sentence"]
        del item1["path"], item1["sentence"]
        resampler = torchaudio.transforms.Resample(orig_freq=item1["sampling_rate"], new_freq=16_000)
        speech_array = resampler.forward(speech_array.squeeze(0)).numpy()
        item1["speech"] = speech_array.tolist()
        item1["sampling_rate"] = 16_000

        if item2 != None:
            speech_array, sampling_rate = torchaudio.load(item2["path"])
            item2["sampling_rate"] = sampling_rate
            item2["target_text"] = item2["sentence"]
            del item2["path"], item2["sentence"]
            resampler = torchaudio.transforms.Resample(orig_freq=item1["sampling_rate"], new_freq=16_000)
            speech_array = resampler.forward(speech_array.squeeze(0)).numpy()
            item2["speech"] = speech_array.tolist()
            item2["sampling_rate"] = 16_000

            item1["speech"] = item1["speech"] + item2["speech"]
            item1["target_text"] = item1["target_text"] + " " + item2["target_text"]

        item1["input_values"] = self.processor(item1["speech"], sampling_rate=item1["sampling_rate"]).input_values[0]

        with self.processor.as_target_processor():
           item1["labels"] = self.processor(item1["target_text"]).input_ids
        del item1["target_text"], item1["speech"], item1["sampling_rate"]

        return item1


print("#### Preparing dataset for training using the CustomDataset")
dataset_train = CustomDatasetConcatenate(dataset = common_voice_train, processor = processor)
dataset_val = CustomDatasetConcatenate(dataset = common_voice_val, processor = processor)




print("#### Initializing training arguments")
training_args = TrainingArguments(
            output_dir='./finetuning_jonata_ita_mlls_aug/checkpoints', # da cambiare
            group_by_length=True,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            eval_accumulation_steps=24,
            evaluation_strategy="steps",
            num_train_epochs=1,
            fp16=False,
            save_steps=100,
            eval_steps=100,
            logging_steps=100,
            learning_rate=3e-4,
            dataloader_num_workers=8,
            warmup_ratio=0.1, #~10% of training
            save_total_limit=1,
            logging_dir='./finetuning_jonata_ita_mlls_aug/logs' # da cambiare
       )

print("#### Creating Trainer")
trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=processor.feature_extractor,
)

try:

    print("#### Start training")
    trainer.train()

    print("#### Save model")
    trainer.save_model('./finetuning_jonata_ita_mlls_aug/final') # cambiare
except Exception as e:
    with open('./finetuning_jonata_ita_mlls_aug/error.txt', 'w') as f: # cambiare
        f.write(str(e))
    print(str(e))
