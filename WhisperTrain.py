import os
import wandb
import torch
import json
import evaluate
from pathlib import Path
import torchaudio
from torch.utils.data import Dataset, ConcatDataset
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import random


class WhisperMultilingualASRDataset(Dataset):
    def __init__(self, json_file, processor=None, apply_augmentation=True):
        self.data = []
        self.language = Path(json_file).stem.split('_')[-1]
        self.apply_augmentation = apply_augmentation
        self.apply_augmentation = apply_augmentation
        if self.apply_augmentation:
            # Whisper uses 128 mel bins, so freq_mask_param should be less than that
            # self.time_stretch = torchaudio.transforms.TimeStretch(n_freq=128, hop_length=160)
            self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)  # about 8% of freq bins
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=30)
        
        # Language mapping for Whisper
        # self.lang_map = {
        #     'en': 'english',
        #     'ru': 'russian',
        #     'kk': 'kazakh',
        #     'tr': 'turkish'
        # }
        
        # Initialize processor if not provided
        if processor is None:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        else:
            self.processor = processor
        
        # Load data
        with open(json_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item['source'])

    def __len__(self):
        return len(self.data)
    
    def time_stretch_audio(self, audio_waveform, sample_rate, stretch_factor):
        """
        Time-stretch an audio waveform using TorchAudio.

        Parameters:
        - audio_waveform (torch.Tensor): The input audio waveform (1D or 2D tensor).
        - sample_rate (int): The sampling rate of the audio (e.g., 16000 Hz).
        - stretch_factor (float): The time-stretch factor (e.g., 1.5 for slower, 0.5 for faster).

        Returns:
        - torch.Tensor: The time-stretched audio waveform.
        """
        # Apply time-stretching with torchaudio.transforms.Resample
        # Convert to spectrogram for stretching
        n_fft = 1024
        hop_length = 160  # Matches 10 ms for 16kHz
        window = torch.hann_window(n_fft)
        stft = torch.stft(audio_waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)

        # Time stretch
        time_stretch = torchaudio.transforms.TimeStretch(n_freq=n_fft // 2 + 1, hop_length=hop_length)
        stretched_stft = time_stretch(stft, stretch_factor)

        # Convert back to waveform
        stretched_audio = torch.istft(stretched_stft, n_fft=n_fft, hop_length=hop_length, window=window)
        return stretched_audio

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            
            # Load audio
            audio_path = "/raid/rakhat_meiramov/projects/asr/21NovData/Data/" + item['audio_local_path']
            speech, sr = torchaudio.load(audio_path)

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                speech = resampler(speech)
                sr = 16000

            rate = 1 / random.choice([0.9, 1.0, 1.1])
            speech = self.time_stretch_audio(audio_waveform=speech, sample_rate=sr, stretch_factor=rate)
            
            # Get features from processor
            features = self.processor.feature_extractor(
                speech[0],
                sampling_rate=16000
            ).input_features[0]

            # Convert to tensor since it's numpy array
            features = torch.from_numpy(features)  # Shape: [128, 3000]


            # Apply augmentations to spectrogram
            if self.apply_augmentation:
                features = self.freq_masking(features)
                features = self.time_masking(features)

            # language = self.lang_map[self.language]
            text = item['text']
            
            prompt_ids = self.processor.tokenizer.get_decoder_prompt_ids(language=self.language, task="transcribe") 
            labels = self.processor.tokenizer(text=text).input_ids 
            
            return {
                "input_features": features,
                "labels": labels
            }
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {e}")
            return None



def calculate_steps_per_epoch(train_dataset, train_batch_size, device_count, gradient_accumulation_steps):
    total_samples = len(train_dataset)
    effective_batch_size = (
        train_batch_size *  # batch size per GPU
        device_count *                  # number of GPUs
        gradient_accumulation_steps    # gradient accumulation
    )
    steps_per_epoch = total_samples // effective_batch_size

    print(f"Total samples: {total_samples}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    return steps_per_epoch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        try:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch
        except Exception as e:
            print(e)
            print(features)

def train_whisper():
    # Specify which GPUs to use
    gpu_ids = [ 4, 5, 6, 7]  # Change these to your desired GPU IDs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    # Initialize process group for DDP
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank == 0:
        wandb.login(key="861ff928f50de914fc850d59133d55bec18996b3")

    # torch.cuda.set_device(local_rank)
    # torch.distributed.init_process_group(
    #     backend="nccl",
    #     init_method="env://"
    # )

    num_steps = 0
    epochs = 5
    batch_size = 8
    gradient_steps = 1

    # Initialize processor and model
    model_name = "openai/whisper-large-v3-turbo"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU
    # model = model.to(local_rank)
    
    # Wrap model with DDP
    # if world_size > 1:
    #     model = DistributedDataParallel(
    #         model, 
    #         device_ids=[local_rank],
    #         output_device=local_rank
    #     )

    # Load datasets
    data_dir = "/raid/rakhat_meiramov/projects/asr/21NovData/Cleaned_JSON"  # Update this path
    train_datasets = []
    eval_datasets = []
    
    languages = ['en', 'ru', 'kk', 'tr']
    # languages = ['en', 'kk']
    
    for lang in languages:
        # Training dataset
        train_file = f"{data_dir}/train_asr_{lang}.json"
        if Path(train_file).exists():
            train_datasets.append(
                WhisperMultilingualASRDataset(train_file, processor=processor, apply_augmentation=True)
            )
            train_datasets.append(
                WhisperMultilingualASRDataset(train_file, processor=processor, apply_augmentation=True)
            )
        
        # Validation dataset
        val_file = f"{data_dir}/dev_asr_{lang}.json"
        if Path(val_file).exists():
            eval_datasets.append(
                WhisperMultilingualASRDataset(val_file, processor=processor, apply_augmentation=False)
            )

    # Combine datasets
    train_dataset = ConcatDataset(train_datasets)
    eval_dataset = ConcatDataset(eval_datasets)
    num_steps = calculate_steps_per_epoch(train_dataset, batch_size, torch.cuda.device_count(), gradient_steps) * epochs
    steps_per_epoch = calculate_steps_per_epoch(train_dataset, batch_size, torch.cuda.device_count(), gradient_steps)
    # num_steps = 63560 + 3 * steps_per_epoch

    if local_rank == 0:
        print(f"Using GPUs: {gpu_ids}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        wandb.init(
            project="whisper-large-v3-turbo-finetuned",
            name="whisper-v3-turbo-finetune",
            config={
                "model_name": "openai/whisper-large-v3-turbo",
                "learning_rate": 5.5e-6,
                "batch_size_per_gpu": batch_size,
                "total_batch_size": batch_size * world_size,
                "grad_accumulation": gradient_steps,
                "languages": languages,
                "warmup_steps": 0.01 * num_steps,
                "max_steps": num_steps,
                "num_gpus": world_size,
                "gpu_ids": gpu_ids
            }
        )
        
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer * 100}

    
    # checkpoint_path = "/raid/rakhat_meiramov/projects/asr/whisper-large-v3-turbo-finetuned/checkpoint-63560"
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-large-v3-turbo-finetuned",
        # resume_from_checkpoint=checkpoint_path,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_steps,
        # gradient_checkpointing=True,
        learning_rate=5.5e-6,
        warmup_ratio=0.01,
        max_steps=num_steps,
        # seed=42,                    # Set random seed for reproducibility
        # data_seed=42,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=steps_per_epoch//4, 
        save_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=steps_per_epoch//4,
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=8,
        eval_on_start=True,
        # Distributed training arguments
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
        ddp_backend="nccl",
        # WandB arguments
        report_to="wandb" if local_rank == 0 else "none",
        run_name=wandb.run.name if local_rank == 0 else None,
    )

    # Define trainer
    trainer = Seq2SeqTrainer(
        # model=model.module if isinstance(model, DistributedDataParallel) else model,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        compute_metrics=compute_metrics,
        processing_class=processor
        # callbacks=[CustomWandbCallback()] if local_rank == 0 else []
    )

    # Train the model
    trainer.train()

    # Save the model and processor (only on main process)
    if local_rank == 0:
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        wandb.finish()

if __name__ == "__main__":
    # Login to wandb (only on main process)
    if int(os.environ.get("LOCAL_RANK", -1)) == 0:
        wandb.login()
    
    train_whisper()