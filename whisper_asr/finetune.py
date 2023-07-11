import json
import os
from pathlib import Path

import evaluate
from transformers.models.whisper import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

from whisper_asr.data import DataCollatorSpeechSeq2SeqWithPadding, prepare_data
from whisper_asr.datatypes import AudioFileCaptionPair

DATA_DIR = Path(os.environ.get("DATA_DIR", "/root/whisper-data"))


def main():
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", task="transcribe")

    with open(DATA_DIR / "audio_caption_pairs.json", "r") as f:
        audio_caption_pairs = [AudioFileCaptionPair.from_dict(x) for x in json.load(f)]

    processor = WhisperProcessor.from_pretrained("openai/whisper-medium", task="transcribe")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    train_pairs = audio_caption_pairs[: int(len(audio_caption_pairs) * 0.8)]
    val_pairs = audio_caption_pairs[int(len(audio_caption_pairs) * 0.8) :]
    train_prepared_dataset = prepare_data(
        train_pairs, feature_extractor, tokenizer, dataset_path=DATA_DIR / "train_dataset"
    )
    val_prepared_dataset = prepare_data(
        val_pairs, feature_extractor, tokenizer, dataset_path=DATA_DIR / "val_dataset"
    )
    metric = evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    model.config.forced_decoder_ids = None  # type: ignore
    model.config.suppress_tokens = []  # type: ignore

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir="./training-checkpoints/whisper-medium",  # change to a repo name of your choice
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        dataloader_num_workers=7,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        # eval_steps=1000,
        logging_steps=25,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_prepared_dataset,
        eval_dataset=val_prepared_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.evaluate(eval_dataset=val_prepared_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
