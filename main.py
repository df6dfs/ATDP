"""Experiment-running framework."""
import os
import pdb
import shutil
import datetime
import yaml
import time
import json
import logging
import sys
import argparse
import importlib
import subprocess
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel
from pytorch_lightning.plugins import DDPPlugin
import nni

from models import RobertaForPrompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    print(f"module_name: {module_name}, class_name: {class_name}")

    module = importlib.import_module(module_name)
    print(f"module: {module}")
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and custom arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer-specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--config", type=str, default="roberta-large")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="DIALOGUE")
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="bert.BertForSequenceClassification")
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--best_model", type=str, default=None)
    parser.add_argument("--knn_topk", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--MVRE", action="store_true", default=False)
    parser.add_argument("--data_type", type=str, default="semeval")
    parser.add_argument("--multi_viewer_num", type=int, default=3)
    parser.add_argument("--pipeline_init", action="store_true", default=False)
    parser.add_argument("--rm_SI", action="store_true", default=False)
    parser.add_argument("--use_contrastive", action="store_true", default=False)
    parser.add_argument("--contrastive_ratio", type=float, default=1.0)
    parser.add_argument("--contrastive_beta", type=float, default=0.5)


    # Experiment workflow arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "average"],
        default="train",
        help="Execution mode: train (training) or average (aggregate results and report to NNI)",
    )
    parser.add_argument("--run_ids", type=int, default=0, help="Run ID for distinguishing trials")
    parser.add_argument("--result_file", type=str, default="./temp_results.json", help="Path to temporary result JSON file")


    # Dynamic argument loading
    temp_args, _ = parser.parse_known_args()

    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    litmodel_class = _import_class(f"lit_models.{temp_args.litmodel_class}")
    print(temp_args.data_class, temp_args.model_class, temp_args.litmodel_class)

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def average_and_report(file_path, args):
    """Read all trial results, compute the average score, and report to NNI."""
    try:
        with open(file_path, "r") as f:
            results_dict = json.load(f)

        all_results = list(results_dict.values())
        if not all_results:
            raise ValueError("No results found to compute average.")

        average_result = sum(all_results) / len(all_results)
        print(f"All trial results: {all_results}")
        print(f"Average result: {average_result:.4f}")

        # Report final aggregated result to NNI
        nni.report_final_result(average_result)
        print("[INFO] Average result successfully reported to NNI.")

    except Exception as e:
        print(f"[ERROR] Exception in average mode: {e}")
        nni.report_final_result(0.0)


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    print(args)

    if args.mode == "train":
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        pl.seed_everything(args.seed)

        data_class = _import_class(f"data.{args.data_class}")
        model_class = _import_class(f"models.{args.model_class}")
        litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")

        config = AutoConfig.from_pretrained(args.config, cache_dir=args.cache_dir)
        model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)

        print(f"Model parameter count: {model.num_parameters() / 1_000_000:.2f} M")

        # Initialize data module
        data = data_class(args, model)
        data_config = data.get_data_config()

        # # Set up dynamic training file
        # orignal_train_name = os.path.join(args.data_dir, "train.txt")
        # filtered_file_name = os.path.join(args.data_dir, "dynamic_filtered_augmented_data.txt")
        # shutil.copy(orignal_train_name, filtered_file_name)
        # print(f"Copied initial training data to: {filtered_file_name}")

        print(data_config)
        print(data.tokenizer)
        model.resize_token_embeddings(len(data.tokenizer))

        lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)
        data.tokenizer.save_pretrained("test")

        logger = pl.loggers.TensorBoardLogger("training/logs")
        dataset_name = args.data_dir.split("/")[-1]
        if args.wandb:
            logger = pl.loggers.WandbLogger(project="dialogue_pl", name=f"{dataset_name}")
            logger.log_hyperparams(vars(args))

        if args.output_dir and not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        early_callback = pl.callbacks.EarlyStopping(
            monitor="Eval/f1", mode="max", patience=40, check_on_train_epoch_end=False
        )
        model_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="Eval/f1",
            mode="max",
            filename="{epoch}-{Eval/f1:.2f}",
            dirpath=args.output_dir,
            save_weights_only=True,
            save_top_k=1,
        )
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        callbacks = [early_callback, model_checkpoint, lr_monitor]

        trainer = pl.Trainer.from_argparse_args(
            args,
            precision=16,
            callbacks=callbacks,
            logger=logger,
            default_root_dir="training/logs",
            gpus=1,
            accelerator=None,
            plugins=None,
        )

        trainer.fit(lit_model, datamodule=data)
        trainer.test(lit_model, datamodule=data)

        # Retrieve best checkpoint path
        best_path = model_checkpoint.best_model_path
        print(f"\nPhase 1 best checkpoint: {best_path}")
        path = best_path

        # Save run configuration YAML
        if not os.path.exists("config"):
            os.mkdir("config")
        config_file_name = time.strftime("%H:%M:%S", time.localtime()) + ".yaml"
        day_name = time.strftime("%Y-%m-%d")
        day_dir = os.path.join("config", day_name)
        if not os.path.exists(day_dir):
            os.mkdir(day_dir)

        config_dict = vars(args)
        config_dict["path"] = path
        with open(os.path.join(day_dir, config_file_name), "w") as file:
            file.write(yaml.dump(config_dict))

        if args.best_model:
            lit_model.load_state_dict(torch.load(args.best_model)["state_dict"])
            print("Successfully loaded lit model from args.best_model.")

        # Clean redundant training and output directories
        print("\n--- Cleaning output and training directories ---")
        da_command = ["rm", "-rf", "output", "training"]
        try:
            subprocess.run(da_command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("[ERROR] Directory cleanup failed with exit code:", e.returncode)
            print("[DEBUG] stderr:\n", e.stderr)

    elif args.mode == "average":
        average_and_report(args.result_file, args)


if __name__ == "__main__":
    main()