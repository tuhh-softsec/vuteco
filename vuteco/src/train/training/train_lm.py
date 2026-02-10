import datetime as dt
import os
import shutil
import sys
from typing import Any, Type, Union

from core.common import global_vars
from core.modeling.modeling_lm_e2e import (LanguageModelE2E,
                                           LanguageModelE2EConfig,
                                           LanguageModelE2EConfigKeys)

if global_vars.MUST_LOAD_UNSLOTH:
    try:
        from unsloth import (FastLanguageModel, is_bfloat16_supported,
                             unsloth_train)
        UNSLOTH_LOADED = True
    except:
        UNSLOTH_LOADED = False
else:
    UNSLOTH_LOADED = False

import numpy as np
import torch
from core.common.augment_data import augment_ds
from core.common.constants import (DETERMINISM, DEVICE, FINAL, RANDOM_SEED,
                                   TEXT_COL)
from core.common.utils_training import (LoggingCallback, clear_up,
                                        compute_performance_clf,
                                        print_stdout_file)
from core.modeling.modeling_lm_common import has_1_in_response
from core.modeling.modeling_lm_fnd import (LanguageModelFinder,
                                           LanguageModelFinderConfig,
                                           LanguageModelFinderConfigKeys)
from core.modeling.modeling_lm_lnk import (LanguageModelLinker,
                                           LanguageModelLinkerConfig,
                                           LanguageModelLinkerConfigKeys)
from datasets import Dataset
from peft import LoraConfig
from transformers import (EvalPrediction, IntervalStrategy, PreTrainedModel,
                          Trainer)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


def preprocess_logits_for_lm(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Convert from shape (batch_size, n_tokens, vocab_size) to (batch_size, n_tokens). Essentially, we get rid of the logits of ALL possible tokens (in the vocabulary, so 30k+) and only get the top predicted next token (with argmax). This vastly reduce the memory footprint, even if we don't use it in compute_metrics.
    return logits.argmax(-1)


def compute_eval_metrics_for_lm(model: Union[LanguageModelFinder, LanguageModelLinker]):
    def _compute_eval_metrics_for_lm(pred: EvalPrediction):
        # expected_labels, expected_tokens = pred.label_ids
        # NOTE LLM - At eval time, we don't have the expected response in the input. So, given how the DataCollatow works, expected_tokens is a list full of -100.
        expected_labels, expected_tokens = pred.label_ids
        #  predicted_tokens = pred.predictions # See preprocess_logits_for_lm for the shape of pred.predictions
        # print(f"Output from Trainer: {predicted_tokens} ({predicted_tokens.shape})")
        predicted_labels = []
        with torch.no_grad():
            for an_input in pred.inputs:
                input_ids = torch.from_numpy(np.expand_dims(an_input, axis=0)).to(model.lm_model.device)
                attention_mask = (input_ids != model.tokenizer.pad_token_id).long().to(model.lm_model.device)
                # NOTE LLM - The prediction in pred.predictions made by Trainer are the teacher-enforced generation, i.e., the output attempts to recreate the input. This is ONLY good during training, not useful during evaluation. Here we call our do_generate() (the same used at inference time) to get what we really want.
                gen_output = model.do_generate(input_ids=input_ids, attention_mask=attention_mask).detach().cpu().numpy().squeeze(0)
                del input_ids, attention_mask
            # for gen_output in generated_outputs:
                # print(f"Generated: {gen_output} -> {repr(model.tokenizer.decode(gen_output))}")
                response = model.extract_responses(gen_output)
                del gen_output
                # print(f"Extracted Response: {response}")
                predicted_labels.append(int(has_1_in_response(response)))
        return compute_performance_clf(expected_labels, predicted_labels)
    return _compute_eval_metrics_for_lm


def run_lm_training(model: Union[LanguageModelFinder, LanguageModelLinker],
                    epochs: int,
                    prep_train_ds: Dataset,
                    prep_eval_ds: Dataset = None,
                    log_outfile: str = None,
                    metric: str = "loss",
                    export_dir: str = None,
                    export_at_end: bool = True) -> tuple[PreTrainedModel, Trainer, dt.timedelta, dict[str, float]]:
    os.makedirs(export_dir, exist_ok=True)
    if model.config.unsloth_training and UNSLOTH_LOADED:
        precision_dict = {
            "fp16": torch.cuda.is_available(),
            #"bf16": is_bfloat16_supported()
        }
    else:
        precision_dict = {
            "fp16": torch.cuda.is_available(),
            #"bf16": torch.cuda.is_bf16_supported()
        }
    training_args = SFTConfig(
        # Training hyperparams
        lr_scheduler_type="linear",
        learning_rate=2e-4,  # 5e-5 for smaller model?
        warmup_ratio=0.1,  # or warmup_steps=10
        weight_decay=0.01,
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8 if torch.cuda.is_available() else 1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_accumulation_steps=8 if torch.cuda.is_available() else None,
        # When to run the evaluation on the evaluation set
        eval_strategy=IntervalStrategy.EPOCH if prep_eval_ds else IntervalStrategy.NO,
        # When to log
        logging_strategy=IntervalStrategy.EPOCH,
        # Log every 10% of the training. By default, if the eval_strategy is STEPS, then eval_steps will be set to logging_steps, so evaluation will also be done every 10% of the training
        # logging_steps=0.1,
        # Extra logging step at the start of the training
        logging_first_step=True,
        # When load_best_model_at_end is True, save_strategy must be like eval_strategy
        save_strategy=IntervalStrategy.EPOCH if prep_eval_ds and export_dir else IntervalStrategy.NO,
        # Keep just the best checkpoint
        save_total_limit=1,
        save_safetensors=False,
        # The metric for which the "best" is determined
        metric_for_best_model=metric,
        # The selected metric will be maximized by default ("True"), unless the metric ends with 'loss' substring"
        # greater_is_better=True,
        # Save a checkpoint every 10% of the training
        # save_steps=0.1,
        # Where to save the checkpoints
        output_dir=export_dir,
        # Load the best checkpoint at the end of the training
        load_best_model_at_end=prep_eval_ds is not None and export_dir is not None,
        # Other config
        disable_tqdm=not sys.stdout.isatty(),
        full_determinism=DETERMINISM,
        seed=RANDOM_SEED if DETERMINISM else False,
        data_seed=RANDOM_SEED if DETERMINISM else False,
        **precision_dict,
        label_names=["label", "labels"],
        include_for_metrics="inputs",  # EvalPrediction object will have the inputs
        # What follows, is the exclusive part to SFT
        max_seq_length=model.model_max_length,
        # Only for newest versions of SFT: max_length=model.model_max_length,
        # dataset_text_field=TEXT_COL,
        # packing=True,
        # dataset_kwargs={
        # "skip_prepare_dataset": True,
        #     "add_special_tokens": False,  # We template with special tokens
        #     "append_concat_token": False,  # No need to add additional separator token
        # },
        # remove_unused_columns=True
    )

    lora_args = {
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "r": 16,  # LoRA rank. Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        "lora_alpha": 16,  # Scaling factor. Usually, alpha=r, but can also be 2*r
        "lora_dropout": 0,  # Unsloth currently only supports dropout = 0
        "bias": "none",  # Unsloth currently only supports bias = "none"
    }
    if model.config.unsloth_training and UNSLOTH_LOADED:
        peft_model = FastLanguageModel.get_peft_model(
            model.lm_model,
            **lora_args,
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context. "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
            random_state=RANDOM_SEED
        )
        train_dict = {
            "model": peft_model,
        }
    else:
        peft_config = LoraConfig(
            **lora_args,
            task_type="CAUSAL_LM",
            # TODO LLM - IDK if these should be added here... maybe not...
            # modules_to_save=["lm_head", "embed_tokens"]
        )
        train_dict = {
            "model": model,
            "peft_config": peft_config,
        }

    # TODO LLM - DEBUG
    #  prep_train_ds = prep_train_ds.select(range(2))
    #  prep_eval_ds = prep_eval_ds.select(range(4))

    # TODO LLM - I don't know if we should temporarily change tokenizer.padding_side to right during training only. Maybe not... let's see later https://github.com/huggingface/transformers/issues/34842
    trainer = SFTTrainer(
        **train_dict,
        args=training_args,
        train_dataset=prep_train_ds,
        eval_dataset=prep_eval_ds,
        processing_class=model.tokenizer,
        # This will tokenize the dataset in the right way to compute the loss only on the part to predict, but it will print some messages about converting to ChatML, applying chat template etc. They must be ignored, as they are printed anyway even if this preprocessing is not happening. It's an SFTTrainer issue
        data_collator=DataCollatorForCompletionOnlyLM(response_template=model.response_starts_after, tokenizer=model.tokenizer),
        # TODO LLM - For now, we don't change anything with the loss function. We left PEFT handle everything. In case we want, we should re-enable the configuration argument again and likely use compute_loss_func (as we're not calling the loss function in forward(), as we did for the NeuralNetwork model family), though we might need to wrap it in another function as the function handler needs num_items_per_batch. See https://github.com/huggingface/transformers/issues/34575.
        # compute_loss_func=model.loss_fct
        compute_metrics=compute_eval_metrics_for_lm(model),
        # During Evaluation, the Trainer stores ALL logit tensors, which might not be needed to compute the evaluation metrics. This function can get rid of things not needed. See https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/14
        preprocess_logits_for_metrics=preprocess_logits_for_lm,
        callbacks=[LoggingCallback(log_outfile)],
    )
    train_start = dt.datetime.now()
    print_stdout_file(f"[{train_start}] Training started!", log_outfile)
    clear_up()
    # NOTE LLM - If I keep getting CUDA out of memory issue at the first evaluation step and if eval_accumulation_step is still not changing anything, I will subclass the Trainer class and override the evaluate() method so that it clear the torch cache before and after calling super().evaluate(). Like this
    """
    def evaluate(self, *args, **kwargs):
        torch.cuda.empty_cache()
        result = super().evaluate(*args, **kwargs)
        torch.cuda.empty_cache()
        return result
    """
    if model.config.unsloth_training and UNSLOTH_LOADED:
        # It should fix gradient accumulation bug (https://unsloth.ai/blog/gradient)
        unsloth_train(trainer)
    else:
        trainer.train()
    clear_up()
    train_end = dt.datetime.now()
    train_duration = train_end - train_start
    print_stdout_file(f"[{train_end}] Training finished! (Total duration: {train_duration})", log_outfile)

    print_stdout_file(f"[{dt.datetime.now()}] Final evaluation started!", log_outfile)
    dev_results = trainer.evaluate(prep_eval_ds) if prep_eval_ds else None
    print_stdout_file(f"[{dt.datetime.now()}] Final evaluation finished!", log_outfile)
    if trainer.state.best_model_checkpoint and os.path.exists(trainer.state.best_model_checkpoint):
        shutil.rmtree(trainer.state.best_model_checkpoint)
    if export_dir and export_at_end:
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(os.path.join(export_dir, FINAL))
        # NOTE We slightly rename adapted_config.json as it would lead from_pretrained to crash, as this JSON has an empty string for base_model_name_or_path, and from_pretrained will try to load from it. We do so also for adapter_model.bin.
        adapter_config_path = os.path.join(export_dir, FINAL, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            os.rename(adapter_config_path, os.path.join(export_dir, FINAL, "_adapter_config.json"))
        adapter_model_path = os.path.join(export_dir, FINAL, "adapter_model.bin")
        if os.path.exists(adapter_model_path):
            os.rename(adapter_model_path, os.path.join(export_dir, FINAL, "_adapter_model.bin"))
        del merged_model
        #trainer.save_model(os.path.join(export_dir, FINAL))
    return model, trainer, train_duration, dev_results


def train_lm_fnd(train_ds: Dataset,
                 model_class: Type[LanguageModelFinder],
                 start_model: LanguageModelFinder = None,
                 model_config: dict[LanguageModelFinderConfigKeys, Any] = None,
                 export_dir: str = None,
                 export_at_end: bool = True,
                 cache_dir: str = None,
                 hyperparam_config: dict[LanguageModelFinderConfigKeys, Any] = None,
                 eval_ds: Dataset = None,
                 metric: str = "loss",
                 log_outfile: str = None) -> tuple[LanguageModelFinder, Trainer, dt.timedelta, dict[str, float]]:
    old_train_size = len(train_ds)
    actual_train_ds = augment_ds(train_ds, model_config[LanguageModelFinderConfigKeys.AUGMENT_TECH], hyperparam_config[LanguageModelFinderConfigKeys.AUGMENT_EXT], log_outfile)
    if len(actual_train_ds) > old_train_size:
        print_stdout_file(f"New training set size: {old_train_size} -> {len(actual_train_ds)}", log_outfile)
    if start_model:
        lm_fnd_model = start_model
    else:
        fnd_config_args = {
            # "loss_fun": model_config[LanguageModelFinderConfigKeys.LOSS],
            # "class_weights": compute_class_weight(class_weight="balanced", classes=np.unique(actual_train_ds[LABEL_COL]), y=actual_train_ds[LABEL_COL]).tolist(),
            LanguageModelFinderConfigKeys.UNSLOTH_TRAINING: model_config[LanguageModelFinderConfigKeys.UNSLOTH_TRAINING],
        }
        lm_fnd_model = model_class(
            LanguageModelFinderConfig(**{k: v for k, v in fnd_config_args.items() if v is not None}),
            cache_dir=cache_dir
        )
    lm_fnd_model.to(DEVICE)
    print_stdout_file(f"Training on device: {DEVICE}", log_outfile)

    print_stdout_file(f"[{dt.datetime.now()}] Preparing training instances", log_outfile)
    prep_actual_train_ds = actual_train_ds.map(lm_fnd_model.prepare_batch_train, batched=True, batch_size=10**3, desc=f"Preparing {len(actual_train_ds)} training instances")
    if eval_ds:
        print_stdout_file(f"[{dt.datetime.now()}] Preparing evaluation instances", log_outfile)
        prep_eval_ds = eval_ds.map(lm_fnd_model.prepare_batch_eval,
                                   batched=True,
                                   batch_size=10**2,
                                   desc=f"Preparing {len(eval_ds)} validation instances",
                                   remove_columns=[c for c in eval_ds.column_names if c in [TEXT_COL]])
    else:
        prep_eval_ds = None
    print_stdout_file(f"[{dt.datetime.now()}] Preparation done!", log_outfile)
    return run_lm_training(model=lm_fnd_model,
                           epochs=hyperparam_config[LanguageModelFinderConfigKeys.EPOCHS],
                           prep_train_ds=prep_actual_train_ds,
                           prep_eval_ds=prep_eval_ds,
                           log_outfile=log_outfile,
                           export_dir=export_dir,
                           metric=metric,
                           export_at_end=export_at_end)


def train_lm_lnk(train_ds: Dataset,
                 model_class: Type[LanguageModelLinker],
                 start_model: LanguageModelLinker = None,
                 model_config: dict[LanguageModelLinkerConfigKeys, Any] = None,
                 export_dir: str = None,
                 export_at_end: bool = True,
                 cache_dir: str = None,
                 hyperparam_config: dict[LanguageModelLinkerConfigKeys, Any] = None,
                 eval_ds: Dataset = None,
                 metric: str = "loss",
                 log_outfile: str = None) -> tuple[LanguageModelLinker, Trainer, dt.timedelta, dict, dict[str, float]]:
    old_train_size = len(train_ds)
    actual_train_ds = augment_ds(train_ds, model_config[LanguageModelLinkerConfigKeys.AUGMENT_TECH], hyperparam_config[LanguageModelLinkerConfigKeys.AUGMENT_EXT], log_outfile)
    if len(actual_train_ds) > old_train_size:
        print_stdout_file(f"New training set size: {old_train_size} -> {len(actual_train_ds)}", log_outfile)
    if start_model:
        lm_lnk_model = start_model
    else:
        lnk_config_args = {
            # "loss_fun": model_config[LanguageModelLinkerConfigKeys.LOSS],
            # "class_weights": compute_class_weight(class_weight="balanced", classes=np.unique(actual_train_ds[LABEL_COL]), y=actual_train_ds[LABEL_COL]).tolist(),
            LanguageModelLinkerConfigKeys.USE_CWE: model_config[LanguageModelLinkerConfigKeys.USE_CWE],
            LanguageModelLinkerConfigKeys.UNSLOTH_TRAINING: model_config[LanguageModelLinkerConfigKeys.UNSLOTH_TRAINING],
        }
        lm_lnk_model = model_class(
            LanguageModelLinkerConfig(**{k: v for k, v in lnk_config_args.items() if v is not None}),
            cache_dir=cache_dir
        )
    lm_lnk_model.to(DEVICE)
    print_stdout_file(f"Training on device: {DEVICE}", log_outfile)

    print_stdout_file(f"[{dt.datetime.now()}] Preparing training instances", log_outfile)
    prep_actual_train_ds = actual_train_ds.map(lm_lnk_model.prepare_batch_train, batched=True, batch_size=10**3, desc=f"Preparing {len(actual_train_ds)} training instances")
    if eval_ds:
        print_stdout_file(f"[{dt.datetime.now()}] Preparing evaluation instances", log_outfile)
        prep_eval_ds = eval_ds.map(lm_lnk_model.prepare_batch_eval,
                                   batched=True,
                                   batch_size=10**2,
                                   desc=f"Preparing {len(eval_ds)} validation instances",
                                   remove_columns=[c for c in eval_ds.column_names if c in [TEXT_COL]])
    else:
        prep_eval_ds = None
    print_stdout_file(f"[{dt.datetime.now()}] Preparation done!", log_outfile)
    return run_lm_training(model=lm_lnk_model,
                           epochs=hyperparam_config[LanguageModelLinkerConfigKeys.EPOCHS],
                           prep_train_ds=prep_actual_train_ds,
                           prep_eval_ds=prep_eval_ds,
                           log_outfile=log_outfile,
                           export_dir=export_dir,
                           metric=metric,
                           export_at_end=export_at_end)


def train_lm_e2e(train_ds: Dataset,
                 model_class: Type[LanguageModelE2E],
                 start_model: LanguageModelE2E = None,
                 model_config: dict[LanguageModelE2EConfigKeys, Any] = None,
                 export_dir: str = None,
                 export_at_end: bool = True,
                 cache_dir: str = None,
                 hyperparam_config: dict[LanguageModelE2EConfigKeys, Any] = None,
                 eval_ds: Dataset = None,
                 metric: str = "loss",
                 log_outfile: str = None) -> tuple[LanguageModelE2E, Trainer, dt.timedelta, dict[str, float]]:
    old_train_size = len(train_ds)
    actual_train_ds = augment_ds(train_ds, model_config[LanguageModelE2EConfigKeys.FT_AUGMENT_TECH], hyperparam_config[LanguageModelE2EConfigKeys.FT_AUGMENT_EXT], log_outfile)
    if len(actual_train_ds) > old_train_size:
        print_stdout_file(f"New training set size: {old_train_size} -> {len(actual_train_ds)}", log_outfile)
    if start_model:
        lm_e2e_model = start_model
    else:
        e2e_config_args = {
            LanguageModelE2EConfigKeys.USE_CWE: model_config[LanguageModelE2EConfigKeys.USE_CWE],
            LanguageModelE2EConfigKeys.UNSLOTH_TRAINING: model_config[LanguageModelE2EConfigKeys.UNSLOTH_TRAINING],
        }
        lm_e2e_model = model_class(
            LanguageModelE2EConfig(**{k: v for k, v in e2e_config_args.items() if v is not None}),
            cache_dir=cache_dir
        )
    lm_e2e_model.to(DEVICE)
    print_stdout_file(f"Training on device: {DEVICE}", log_outfile)

    print_stdout_file(f"[{dt.datetime.now()}] Preparing training instances", log_outfile)
    prep_actual_train_ds = actual_train_ds.map(lm_e2e_model.prepare_batch_train, batched=True, batch_size=10**3, desc=f"Preparing {len(actual_train_ds)} training instances")
    if eval_ds:
        print_stdout_file(f"[{dt.datetime.now()}] Preparing evaluation instances", log_outfile)
        prep_eval_ds = eval_ds.map(lm_e2e_model.prepare_batch_eval,
                                   batched=True,
                                   batch_size=10**2,
                                   desc=f"Preparing {len(eval_ds)} validation instances",
                                   remove_columns=[c for c in eval_ds.column_names if c in [TEXT_COL]])
    else:
        prep_eval_ds = None
    print_stdout_file(f"[{dt.datetime.now()}] Preparation done!", log_outfile)
    return run_lm_training(model=lm_e2e_model,
                           epochs=hyperparam_config[LanguageModelE2EConfigKeys.FT_EPOCHS],
                           prep_train_ds=prep_actual_train_ds,
                           prep_eval_ds=prep_eval_ds,
                           log_outfile=log_outfile,
                           export_dir=export_dir,
                           metric=metric,
                           export_at_end=export_at_end)
