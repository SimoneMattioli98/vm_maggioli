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
import csv
import time
import torchaudio
from torch import nn
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
import collections
from datasets import Dataset, load_dataset, load_metric, concatenate_datasets
import argparse
from os import listdir, walk
from os.path import isfile, join
import IPython.display as ipd
import numpy as np
import random
import librosa
import numpy as np

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


class CTCTrainer(Trainer):

    def __init__(self, length_field_name="length", upload_model_to_wandb_each_step=None, lr_warmup_ratio=0.1, 
                lr_constant_ratio=0.4, sampling_rate=16_000, **kwargs):
        super().__init__(**kwargs)
        self.length_field_name = length_field_name
        self.upload_model_to_wandb_each_step = upload_model_to_wandb_each_step
        self.lr_warmup_ratio = lr_warmup_ratio
        self.lr_constant_ratio = lr_constant_ratio
        self.sampling_rate = sampling_rate

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            lengths = self.train_dataset[self.length_field_name] if self.length_field_name is not None else None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset, self.args.train_batch_size, lengths=lengths, model_input_name=model_input_name
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                )

        else:
            return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.
        This method was built based on https://arxiv.org/pdf/2006.13979 :
            "The learning rate schedule has three phases: warm up for the first 10% of updates, 
             keep constant for 40% and then linearly decay for the remainder"
        
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        def lr_lambda(current_step):
            warmup_steps = int(num_training_steps * self.lr_warmup_ratio)
            constant_steps = int(num_training_steps * self.lr_constant_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif (self.lr_warmup_ratio + self.lr_constant_ratio) == 1.0 or current_step < (warmup_steps + constant_steps):
                return 1
            else: 
                return max(
                    0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (warmup_steps + constant_steps)))
                )
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _apply_some_audio_transformations(self, inputs):
        """Perform some audio transformations"""
        
        # adding an extra dimmention for the channels as our data is mono audio and
        # the expected shape of input for torch_audiomentations is (batch_size, num_channels, num_samples)
        transformed_inputs = inputs["input_values"].unsqueeze(1)

        transformed_inputs = self.augmentator(transformed_inputs, sample_rate=self.sampling_rate)
           
        # returning the inputs to the original shape
        transformed_inputs = torch.squeeze(transformed_inputs, 1)
        
        inputs["input_values"] = transformed_inputs

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

    
        return loss.detach()



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


# def extract_all_chars(batch):
#     all_text = " ".join(batch)
#     vocab = list(set(all_text))
#     return {"vocab": [vocab], "all_text": [all_text]}

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


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



common_voice_train = create_new_cmmv(f"{fold}train2.csv")
common_voice_val = create_new_cmmv(f"{fold}dev2.csv")


CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", ";", ":", '""', "%", '"', "?", "?", "·", "?", "~", "?",
                   "?", "?", "?", "?", "«", "»", "„", "“", "”", "?", "?", "‘", "’", "«", "»", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "?", "‹", "›", "©", "®", "—", "?", "?",
                   "?", "?", "?", "?", "~", "?", ",", "{", "}", "(", ")", "[", "]", "?", "?", "?", "?",
                   "?", "?", "?", "?", "?", "?", "?", ":", "!", "?", "?", "?", "/", "\\", "º", "-", "^", "?", "ˆ"]

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"


def remove_special_characters_comm(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).strip().upper() + " "
    return batch


common_voice_train = common_voice_train.map(remove_special_characters_comm)
common_voice_val = common_voice_val.map(remove_special_characters_comm)

print(common_voice_train)
print(common_voice_val)

#print("#### Creating vocabolary")
#all_text = " ".join([v for k, v in labels_dict_train.items()])
#vocab = list(set(all_text))
#vocab_train = {"vocab": [vocab], "all_text": [all_text]}
#all_text = " ".join([v for k, v in labels_dict_val.items()])
#vocab = list(set(all_text))
#vocab_val = {"vocab": [vocab], "all_text": [all_text]}


#vocab_list = list(set(vocab_train['vocab'][0]) | set(vocab_val["vocab"][0]))

#vocab_dict = {v: k for k, v in enumerate(vocab_list)}

#print(vocab_dict)



#vocab_dict["|"] = vocab_dict[" "]
#del vocab_dict[" "]
#vocab_dict["[UNK]"] = len(vocab_dict)
#vocab_dict["[PAD]"] = len(vocab_dict)
#print(len(vocab_dict))
#print(vocab_dict)


#with open('./finetuning/vocab.json', 'w') as vocab_file:
#    json.dump(vocab_dict, vocab_file)


#print("#### Creating Tokenizer, Feature extractor and Processor")
#tokenizer = Wav2Vec2CTCTokenizer("./finetuning_fb_ita_comm_mlls/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")


#feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)


#processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


#processor.save_pretrained('./finetuning_fb_ita_comm_mlls/processor')



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
        speech_array, sampling_rate = torchaudio.load(f"{fold}clips2/{item['path']}")
        item["speech"] = speech_array[0].numpy()
        item["sampling_rate"] = sampling_rate
        item["target_text"] = item["sentence"]
        del item["path"], item["sentence"]
        item["speech"] = librosa.resample(np.asarray(item["speech"]), sampling_rate, 16_000)
        item["sampling_rate"] = 16_000
        item["input_values"] = self.processor(item["speech"], sampling_rate=item["sampling_rate"]).input_values[0]

        with self.processor.as_target_processor():
           item["labels"] = self.processor(item["target_text"]).input_ids
        del item["target_text"], item["speech"], item["sampling_rate"]

        item["length"] = len(item["input_values"])

        return item




print("#### Preparing dataset for training using the CustomDataset")
dataset_train = CustomDataset(dataset = common_voice_train, processor = processor)
dataset_val = CustomDataset(dataset = common_voice_val, processor = processor)




print("#### Initializing training arguments")
training_args = TrainingArguments(
            output_dir='./finetuning_jonatas_new_cmmv/checkpoints',
            #group_by_length=True,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
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
       )

print("#### Creating Trainer")
trainer = CTCTrainer(
        model=model,
        #length_field_name="length",
        lr_warmup_ratio = 0.1,
        lr_constant_ratio = 0.4,
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
    trainer.save_model('./finetuning_jonatas_new_cmmv/final')
except Exception as e:
    with open('./finetuning_jonatas_new_cmmv/error.txt', 'w') as f:
        f.write(str(e))
    print(str(e))

