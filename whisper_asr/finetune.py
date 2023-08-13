import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import click
import evaluate
import torch
from torch import argmax, nn, softmax
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

import wandb
from whisper_asr.configs import Models
from whisper_asr.data import (
    DataCollatorSpeechSeq2SeqWithPadding,
    DataCollatorWithBatchNumber,
    prepare_data,
)
from whisper_asr.datatypes import AudioFileCaptionPair

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# for k

DATA_DIR = Path(os.environ.get("DATA_DIR", "/root/whisper-data"))


class CustomTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Override self.model.generation_config if a GenerationConfig is specified in args.
        # Priority: args.generation_config > model.generation_config > default GenerationConfig.
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config
        self._initalize_gen_kwargs()

    def _initalize_gen_kwargs(self, **gen_kwargs):
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.args.generation_num_beams
        )
        logger.info(f"Setting generation kwargs: {gen_kwargs}")
        self._gen_kwargs = gen_kwargs

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        batch_number = inputs.pop("batch_number")
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            if batch_number % 50 == 0:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                output_ids = torch.argmax(torch.softmax(outputs["logits"], -1), -1)
                prediction_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                batch_labels = self.tokenizer.batch_decode(
                    inputs["labels"], skip_special_tokens=True
                )
                wer_per_sample = (
                    torch.sum(output_ids != inputs["labels"], dim=-1).float()
                    / torch.sum(inputs["labels"] != -100, dim=-1).float()
                )
                # FIXME: Gradient accumulation fks this up
                wandb.log(
                    {
                        "TrainPredictions": wandb.Table(
                            columns=["predictions", "ground_truths", "wer", "batch_number"],
                            data=list(
                                zip(
                                    prediction_texts,
                                    batch_labels,
                                    wer_per_sample,
                                    [batch_number] * len(prediction_texts),
                                )
                            ),
                        ),
                    }
                )
            else:
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()  # type: ignore
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: List[str] | None = None,
    ) -> Tuple[float | None, torch.Tensor | None, torch.Tensor | None]:
        if ignore_keys:
            for k in ignore_keys:
                inputs.pop(k, None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


@click.command()
@click.option(
    "model_name",
    "--model",
    "-m",
    type=click.Choice([c.name for c in Models]),
    default=Models.WHISPER_TINY.name,
    help="Model name",
)
def main(model_name: str = Models.WHISPER_TINY.value):
    model_name = Models[model_name].value
    # feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    # tokenizer = WhisperTokenizer.from_pretrained(model_name, task="transcribe")

    with open(DATA_DIR / "audio_caption_pairs.json", "r") as f:
        audio_caption_pairs = [AudioFileCaptionPair.from_dict(x) for x in json.load(f)]

    processor = WhisperProcessor.from_pretrained(model_name, task="transcribe")
    data_collator = DataCollatorWithBatchNumber(processor=processor)
    # data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    # DataCollatorWithBatchNumber
    train_pairs = audio_caption_pairs[: int(len(audio_caption_pairs) * 0.8)]
    val_pairs = audio_caption_pairs[int(len(audio_caption_pairs) * 0.8) :]
    train_prepared_dataset = prepare_data(
        train_pairs, processor.feature_extractor, processor.tokenizer, dataset_path=DATA_DIR / "train_dataset"  # type: ignore
    )
    val_prepared_dataset = prepare_data(
        val_pairs, processor.feature_extractor, processor.tokenizer, dataset_path=DATA_DIR / "val_dataset"  # type: ignore
    )
    metric = evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None  # type: ignore
    model.config.suppress_tokens = []  # type: ignore

    def compute_metrics(pred: EvalLoopOutput):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id  # type: ignore

        # we do not want to group tokens when computing the metrics
        pred_str: List[str] = [
            x.lower().strip()
            for x in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)  # type: ignore
        ]
        # pred_str = [" " if x == "" else x for x in pred_str]

        label_str: List[str] = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)  # type: ignore

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)  # type: ignore
        # for pred, label in zip(pred_str[:5], label_str[:5]):
        # print(f"pred: {pred}")
        # print(f"label: {label}")
        # print()

        wandb.log(
            {
                "ValPredictions": wandb.Table(
                    columns=["predictions", "ground_truths"],
                    data=list(zip(pred_str, label_str)),
                ),
            }
        )

        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir="./training-checkpoints/whisper-medium",  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        dataloader_num_workers=8,
        gradient_checkpointing=False,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=255,
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

    trainer = CustomTrainer(
        args=training_args,
        model=model,  # type: ignore
        train_dataset=train_prepared_dataset,  # type: ignore
        eval_dataset=val_prepared_dataset,  # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,  # type: ignore
    )
    wandb.init(project="whisper", config=training_args.to_dict())
    trainer.evaluate(eval_dataset=val_prepared_dataset, ignore_keys=["batch_number"])  # type: ignore
    trainer.train(
        ignore_keys_for_eval=["batch_number"],
    )


if __name__ == "__main__":
    main()
