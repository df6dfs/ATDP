import os
import json
import random
from pathlib import Path
from functools import reduce, partial
from copy import deepcopy

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from .base import BaseLitModel
from .util import f1_eval, compute_f1, acc, f1_score


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss implementation.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Optional: per-class weighting tensor
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [batch_size, num_classes]
        # targets: [batch_size]

        # 1. Compute standard cross entropy loss (retaining per-sample loss)
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)

        # 2. Derive p_t: since ce_loss = -log(p_t), p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # 3. Compute Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Return loss based on the reduction strategy
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def decode(tokenizer, output_ids):
    return [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for g in output_ids
    ]


class BertLitModel(BaseLitModel):
    """
    Uses AutoModelForMaskedLM and selects output representations via auxiliary prediction heads.
    """

    def __init__(self, model, args, tokenizer):
        super().__init__(model, args)
        self.tokenizer = tokenizer

        with open(f"{args.data_dir}/rel2id.json", "r") as file:
            rel2id = json.load(file)

        na_num = 0
        self.id2rel = {}
        for k, v in rel2id.items():
            if k in ["NA", "no_relation", "Other"]:
                na_num = v
                break
        self.na_num = int(na_num)
        self.rel2id = rel2id
        for k, v in rel2id.items():
            self.id2rel[int(v)] = k
        num_relation = len(rel2id)

        # Initialize loss function
        if getattr(args, "use_focal_loss", 0) == 1 and "dialogue" not in args.data_dir:
            self.loss_fn = FocalLoss(gamma=getattr(args, "focal_gamma", 2.0))
        else:
            self.loss_fn = (
                multilabel_categorical_crossentropy
                if "dialogue" in args.data_dir
                else nn.CrossEntropyLoss()
            )

        # Evaluation function ignoring NA/no_relation
        self.eval_fn = (
            f1_eval
            if "dialogue" in args.data_dir
            else partial(f1_score, rel_num=num_relation, na_num=na_num)
        )
        self.best_f1 = 0
        self.t_lambda = args.t_lambda

        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)["input_ids"][0]
        self.relation_tag = tokenizer("-", add_special_tokens=False)["input_ids"][0]
        self.relation_tokens = []
        self.subject_word = []
        self.object_word = []
        self.final_word = []
        self.other_subject_word = None
        self.other_object_word = None

        self._init_label_word()
        self.log_vars = nn.Parameter(torch.zeros(2))

    def _init_label_word(self):
        args = self.args
        dataset_name = args.data_type
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt"

        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)
        self.label_word_idx = label_word_idx
        num_labels = len(label_word_idx)

        for a in range(1, num_labels + 1):
            if args.MVRE:
                for j in range(args.multi_viewer_num):
                    self.tokenizer.add_tokens(f"[class{a}_{j}]", special_tokens=True)
            self.tokenizer.add_tokens(f"[relation{a}]", special_tokens=True)
        self.tokenizer.add_tokens("[other_subject_word]", special_tokens=True)
        self.tokenizer.add_tokens("[other_word]", special_tokens=True)

        # Inject virtual tokens for Task 4
        self.tokenizer.add_tokens(["[v_highly]", "[v_not]"], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

        orig_highly_id = self.tokenizer.encode(" highly", add_special_tokens=False)[0]
        orig_not_id = self.tokenizer.encode(" not", add_special_tokens=False)[0]

        if getattr(self.args, "multi_task_lesson_on", 1) == 1:
            from transformers import pipeline
            from collections import Counter

            unmasker = pipeline("fill-mask", model="./models/roberta-large", device=-1)
            mask_token = unmasker.tokenizer.mask_token

            sample_file = f"./template/{self.args.data_type}.txt"
            pos_top_tokens, neg_top_tokens = [], []

            with open(sample_file, "r", encoding="utf-8") as f:
                for line in f:
                    data_i = json.loads(line)
                    relation = data_i["relation"]
                    str_i = (
                        " ".join(data_i["token"])
                        + " "
                        + "".join(data_i["h"]["name"])
                        + " and "
                        + "".join(data_i["t"]["name"])
                        + f" are {mask_token} related."
                    )

                    output = unmasker(str_i, top_k=1)
                    top_token_str = output[0]["token_str"]
                    top_token_id = self.tokenizer.encode(top_token_str, add_special_tokens=False)[0]

                    if relation in ["NA", "no_relation", "Other"]:
                        neg_top_tokens.append(top_token_id)
                    else:
                        pos_top_tokens.append(top_token_id)

            best_pos_id = (
                Counter(pos_top_tokens).most_common(1)[0][0]
                if pos_top_tokens
                else orig_highly_id
            )
            best_neg_id = (
                Counter(neg_top_tokens).most_common(1)[0][0]
                if neg_top_tokens
                else orig_not_id
            )

            with torch.no_grad():
                v_highly_id = self.tokenizer.encode("[v_highly]", add_special_tokens=False)[0]
                v_not_id = self.tokenizer.encode("[v_not]", add_special_tokens=False)[0]
                word_embeddings = self.model.get_input_embeddings()

                word_embeddings.weight[v_highly_id] = torch.mean(
                    torch.stack([word_embeddings.weight[orig_highly_id], word_embeddings.weight[best_pos_id]]),
                    dim=0,
                )
                word_embeddings.weight[v_not_id] = torch.mean(
                    torch.stack([word_embeddings.weight[orig_not_id], word_embeddings.weight[best_neg_id]]),
                    dim=0,
                )

            if hasattr(self.model, "tie_weights"):
                self.model.tie_weights()

        self.predict = nn.ModuleList(
            [
                nn.Linear(self.model.config.hidden_size, 1).to(self.device)
                for _ in range(args.multi_viewer_num)
            ]
        )

        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            if args.MVRE:
                continous_label_word = [
                    [
                        self.tokenizer(f"[class{i}_{j}]", add_special_tokens=False)["input_ids"]
                        for j in range(args.multi_viewer_num)
                    ]
                    for i in range(1, num_labels + 1)
                ]
            else:
                continous_label_word = [
                    a[0]
                    for a in self.tokenizer(
                        [f"[class{i}]" for i in range(1, num_labels + 1)],
                        add_special_tokens=False,
                    )["input_ids"]
                ]

            if self.args.init_answer_words:
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    sample_file = "./template/semeval.txt"
                    if self.args.data_type == "tacrev":
                        sample_file = "./template/tacrev.txt"
                    elif self.args.data_type == "tacred":
                        sample_file = "./template/tacred.txt"

                    data = {}
                    with open(sample_file, "r") as f:
                        for line in f:
                            data_i = json.loads(line)
                            str_i = (
                                " ".join(data_i["token"])
                                + "".join(data_i["h"]["name"])
                                + " "
                                + "<mask> " * args.multi_viewer_num
                                + "".join(data_i["t"]["name"])
                            )
                            reverse_str_i = (
                                " ".join(data_i["token"])
                                + "".join(data_i["t"]["name"])
                                + " "
                                + "<mask> " * args.multi_viewer_num
                                + "".join(data_i["h"]["name"])
                            )
                            rel_id_str = str(self.rel2id[data_i["relation"]])
                            if rel_id_str not in data:
                                data[rel_id_str] = []
                            data[rel_id_str].append(str_i)
                            data[rel_id_str].append(reverse_str_i)

                    from transformers import pipeline

                    unmasker = pipeline("fill-mask", model="./models/roberta-large")
                    for i, idx in enumerate(label_word_idx):
                        if args.MVRE:
                            str_test = data.get(str(i), [None])[0]
                            output = unmasker(str_test) if str_test is not None else None

                            for j in range(args.multi_viewer_num):
                                if not self.args.pipeline_init:
                                    word_embeddings.weight[continous_label_word[i][j]] = torch.mean(
                                        word_embeddings.weight[idx], dim=0
                                    )
                                    continue
                                if str_test is None:
                                    word_embeddings.weight[continous_label_word[i][j]] = torch.mean(
                                        word_embeddings.weight[idx], dim=0
                                    )
                                else:
                                    token_str = (
                                        output[0]["token_str"]
                                        if args.multi_viewer_num == 1
                                        else output[j][0]["token_str"]
                                    )
                                    token_list = [self.tokenizer.encode(token_str)[1]]
                                    if self.args.rm_SI:
                                        word_embeddings.weight[continous_label_word[i][j]] = torch.mean(
                                            word_embeddings.weight[token_list], dim=0
                                        )
                                    else:
                                        combined_tokens = torch.tensor(token_list + idx.numpy().tolist())
                                        word_embeddings.weight[continous_label_word[i][j]] = torch.mean(
                                            word_embeddings.weight[combined_tokens], dim=0
                                        )
                        else:
                            word_embeddings.weight[continous_label_word[i]] = torch.mean(
                                word_embeddings.weight[idx], dim=0
                            )

            if self.args.init_type_words:
                so_word = [
                    a[0]
                    for a in self.tokenizer(["[obj]", "[sub]"], add_special_tokens=False)["input_ids"]
                ]
                meaning_word = [
                    a[0]
                    for a in self.tokenizer(
                        ["person", "organization", "location", "date", "country"],
                        add_special_tokens=False,
                    )["input_ids"]
                ]
                for i in range(len(so_word)):
                    word_embeddings.weight[so_word[i]] = torch.mean(
                        word_embeddings.weight[meaning_word], dim=0
                    )

            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)

        if args.MVRE:
            continous_label_word = []
            self.relation_tokens = [
                [
                    self.tokenizer(f"[class{i}_{j}]", add_special_tokens=False)["input_ids"]
                    for j in range(args.multi_viewer_num)
                ]
                for i in range(1, num_labels + 1)
            ]
            for i in range(len(label_word_idx)):
                continous_label_word.append([self.relation_tokens[i][j] for j in range(args.multi_viewer_num)])

        self.word2label = continous_label_word

    def forward(self, x):
        return self.model(x)

    def fusion_sum_mask(self, input_ids, hidden_state):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        subject_mask_idx = mask_idx[[i for i in range(0, len(mask_idx), 3)]]
        relation_mask_idx = mask_idx[[i for i in range(1, len(mask_idx), 3)]]
        object_mask_idx = mask_idx[[i for i in range(2, len(mask_idx), 3)]]
        bs = input_ids.shape[0]

        subject_hidden_state = hidden_state[torch.arange(bs), subject_mask_idx]
        relation_hidden_state = hidden_state[torch.arange(bs), relation_mask_idx]
        object_hidden_state = hidden_state[torch.arange(bs), object_mask_idx]

        sum_mask_hidden_state = subject_hidden_state + relation_hidden_state + object_hidden_state
        logits = self.model.lm_head(sum_mask_hidden_state)
        return logits[:, self.final_word]

    def training_step(self, batch, batch_idx):
        (
            main_input_ids,
            main_attention_mask,
            task2_input_ids,
            task2_mlm_label,
            task3_input_ids,
            task3_mlm_label,
            task4_input_ids,
            task4_attention_mask,
            task4_mlm_label,
            labels,
        ) = batch

        # Curriculum learning weight scheduling
        progress_ratio = (self.current_epoch + 1) / float(self.trainer.max_epochs)
        mlm_weight = progress_ratio * self.args.class_learning_threshold
        re_weight = 1.0 - mlm_weight

        loss_fct = nn.CrossEntropyLoss()

        # Task 0: Main relation extraction task
        result_main = self.model(
            main_input_ids, main_attention_mask, return_dict=True, output_hidden_states=True
        )
        logits_main = result_main.logits

        if self.args.MVRE:
            logits1, _ = self.pvp_multi_viewer(
                logits_main, main_input_ids, hidden_state=result_main.hidden_states[-1]
            )
            loss_re = self.loss_fn(logits1, labels)

            if self.args.use_contrastive_loss:
                contrastive_loss = self.contrastive_loss(beta=self.args.contrastive_beta)
                loss_re = loss_re + self.args.contrastive_ratio * contrastive_loss
        else:
            logits_main, _ = self.pvp(logits_main, main_input_ids, labels)
            loss_re = self.loss_fn(logits_main, labels)

        self.log("Train/re_loss", loss_re)

        if self.args.multi_task_lesson_on:
            # Task 2: Subject MLM prediction
            result_t2 = self.model(input_ids=task2_input_ids, attention_mask=main_attention_mask, return_dict=True)
            loss_mlm2 = loss_fct(result_t2.logits.view(-1, self.model.config.vocab_size), task2_mlm_label.view(-1))

            # Task 3: Object MLM prediction
            result_t3 = self.model(input_ids=task3_input_ids, attention_mask=main_attention_mask, return_dict=True)
            loss_mlm3 = loss_fct(result_t3.logits.view(-1, self.model.config.vocab_size), task3_mlm_label.view(-1))

            # Task 4: Relation existence detection (binary classification over virtual tokens)
            result_t4 = self.model(input_ids=task4_input_ids, attention_mask=task4_attention_mask, return_dict=True)
            logits_t4 = result_t4.logits

            mask_positions = task4_mlm_label != -100
            active_logits_t4 = logits_t4[mask_positions]
            active_labels_t4 = task4_mlm_label[mask_positions]

            highly_id = self.tokenizer.encode("[v_highly]", add_special_tokens=False)[0]
            not_id = self.tokenizer.encode("[v_not]", add_special_tokens=False)[0]
            candidate_logits_t4 = active_logits_t4[:, [highly_id, not_id]]

            orig_highly_id = self.tokenizer.encode(" highly", add_special_tokens=False)[0]
            binary_labels_t4 = torch.where(active_labels_t4 == orig_highly_id, 0, 1)

            loss_mlm4 = self.loss_fn(candidate_logits_t4, binary_labels_t4)
            loss_mlm = (loss_mlm2 + loss_mlm3 + loss_mlm4) / 3.0

            self.log("Train/task4_loss", loss_mlm4)
            self.log("Train/mlm_loss", loss_mlm)

            total_loss = re_weight * loss_re + mlm_weight * loss_mlm
        else:
            total_loss = loss_re

        self.log("Train/total_loss", total_loss)
        return total_loss

    def get_loss(self, logits, input_ids, labels):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        return self.loss_fn(mask_output, labels)

    def validation_step(self, batch, batch_idx):
        (
            main_input_ids,
            main_attention_mask,
            task2_input_ids,
            task2_mlm_label,
            task3_input_ids,
            task3_mlm_label,
            task4_input_ids,
            task4_attention_mask,
            task4_mlm_label,
            labels,
        ) = batch

        result = self.model(main_input_ids, main_attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits

        if self.args.MVRE:
            if self.args.merge_to_one:
                logits_list, _ = self.pvp_one(logits, main_input_ids, hidden_state=result.hidden_states[-1])
                logits_merge = torch.cat(logits_list, dim=1)
                logits1 = torch.sum(logits_merge, dim=1)
            else:
                logits1, _ = self.pvp_multi_viewer(logits, main_input_ids, hidden_state=result.hidden_states[-1])
        else:
            logits1, _ = self.pvp(logits, main_input_ids, hidden_state=result.hidden_states[-1])

        logits = logits1
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)

        prob_not_np = np.array([])
        if self.args.multi_task_lesson_on:
            with torch.no_grad():
                result_t4 = self.model(input_ids=task4_input_ids, attention_mask=task4_attention_mask, return_dict=True)
                logits_t4 = result_t4.logits

                mask_positions = task4_mlm_label != -100
                active_logits = logits_t4[mask_positions]

                highly_id = self.tokenizer.encode("[v_highly]", add_special_tokens=False)[0]
                not_id = self.tokenizer.encode("[v_not]", add_special_tokens=False)[0]

                candidate_logits = active_logits[:, [highly_id, not_id]]
                candidate_probs = torch.softmax(candidate_logits, dim=-1)
                prob_not_np = candidate_probs[:, 1].detach().cpu().numpy()

        return {
            "eval_logits": logits.detach().cpu().numpy(),
            "eval_labels": labels.detach().cpu().numpy(),
            "inputs": main_input_ids.detach().cpu().numpy(),
            "task4_prob_not": prob_not_np,
        }

    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        # Calculate class-adaptive rejection thresholds
        self.class_thresholds = {}
        if getattr(self.args, "use_adaptive_threshold", 1) == 1:
            probs = torch.softmax(torch.tensor(logits).float(), dim=-1).numpy()
            for c in range(len(self.id2rel)):
                if c == self.na_num:
                    continue
                mask = labels == c
                if np.sum(mask) > 0:
                    class_probs = probs[mask, c]
                    mu = np.mean(class_probs)
                    sigma = np.std(class_probs)
                    threshold = mu - self.args.reject_lambda * sigma
                    self.class_thresholds[c] = max(0.05, threshold)
                else:
                    self.class_thresholds[c] = 0.2

        # Task 4 soft veto threshold search and application
        if getattr(self.args, "multi_task_lesson_on", 0) == 1 and getattr(self.args, "use_task4_veto", 1) == 1:
            if "task4_prob_not" in outputs[0]:
                t4_prob_not = np.concatenate([o["task4_prob_not"] for o in outputs])

                if len(t4_prob_not) == len(logits):
                    candidate_thresholds = [
                        0.0001, 0.0003, 0.0005, 0.0007, 0.0009,
                        0.001, 0.003, 0.005, 0.007, 0.009,
                        0.01, 0.03, 0.05, 0.07, 0.09, 0.1,
                        0.3, 0.5, 0.7, 0.9,
                    ]
                    best_t = 0.5
                    best_search_f1 = 0.0
                    base_logits = logits.copy()

                    for t in candidate_thresholds:
                        temp_logits = base_logits.copy()
                        veto_mask = t4_prob_not > t
                        temp_logits[veto_mask, :] = -1e7
                        temp_logits[veto_mask, self.na_num] = 1e7

                        temp_metrics = self.eval_fn(temp_logits, labels)
                        if temp_metrics["f1"] > best_search_f1:
                            best_search_f1 = temp_metrics["f1"]
                            best_t = t

                    self.args.task4_veto_threshold = best_t

                    final_veto_mask = t4_prob_not > best_t
                    logits[final_veto_mask, :] = -1e7
                    logits[final_veto_mask, self.na_num] = 1e7

        f1 = self.eval_fn(logits, labels)["f1"]
        self.log("Eval/f1", f1)
        print(f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, on_epoch=True,prog_bar=True)

    def test_step(self, batch, batch_idx):
        (
            main_input_ids,
            main_attention_mask,
            task2_input_ids,
            task2_mlm_label,
            task3_input_ids,
            task3_mlm_label,
            task4_input_ids,
            task4_attetion_mask,
            task4_mlm_label,
            labels,
        ) = batch

        result = self.model(main_input_ids, main_attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits

        if self.args.MVRE:
            if self.args.merge_to_one:
                logits_list, _ = self.pvp_one(logits, main_input_ids, hidden_state=result.hidden_states[-1])
                logits_merge = torch.cat(logits_list, dim=1)
                logits1 = torch.max(logits_merge, dim=1)[0]
            else:
                logits1, _ = self.pvp_multi_viewer(logits, main_input_ids, hidden_state=result.hidden_states[-1])
        else:
            logits1, _ = self.pvp(logits, main_input_ids, hidden_state=result.hidden_states[-1])

        logits = logits1

        task4_prob_not_np = np.array([])
        if getattr(self.args, "multi_task_lesson_on", 0) == 1 and getattr(self.args, "use_task4_veto", 1) == 1:
            result_t4 = self.model(input_ids=task4_input_ids, attention_mask=task4_attetion_mask, return_dict=True)
            logits_t4 = result_t4.logits

            mask_positions = task4_mlm_label != -100
            active_logits = logits_t4[mask_positions]

            highly_id = self.tokenizer.encode("[v_highly]", add_special_tokens=False)[0]
            not_id = self.tokenizer.encode("[v_not]", add_special_tokens=False)[0]

            candidate_logits = active_logits[:, [highly_id, not_id]]
            candidate_probs = torch.softmax(candidate_logits, dim=-1)
            task4_prob_not_np = candidate_probs[:, 1].detach().cpu().numpy()

        return {
            "test_logits": logits.detach().cpu().numpy(),
            "test_labels": labels.detach().cpu().numpy(),
            "task4_prob_not": task4_prob_not_np,
            "inputs": main_input_ids.detach().cpu().numpy(),
        }

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        # Apply Task 4 soft veto mechanism
        if getattr(self.args, "multi_task_lesson_on", 0) == 1 and getattr(self.args, "use_task4_veto", 1) == 1:
            if "task4_prob_not" in outputs[0] and len(outputs[0]["task4_prob_not"]) > 0:
                t4_prob_not = np.concatenate([o["task4_prob_not"] for o in outputs])
                threshold = getattr(self.args, "task4_veto_threshold", 0.5)

                if len(t4_prob_not) == len(logits):
                    veto_mask = t4_prob_not > threshold
                    if np.sum(veto_mask) > 0:
                        logits[veto_mask, :] = -1e7
                        logits[veto_mask, self.na_num] = 1e7

        # Apply class-wise adaptive threshold rejection
        probs = torch.softmax(torch.tensor(logits).float(), dim=-1).numpy()
        preds = np.argmax(logits, axis=1)

        if getattr(self.args, "use_adaptive_threshold", 1) == 1:
            for i in range(len(preds)):
                pred_class = preds[i]
                if pred_class != self.na_num and hasattr(self, "class_thresholds"):
                    if pred_class in self.class_thresholds:
                        if probs[i, pred_class] < self.class_thresholds[pred_class]:
                            logits[i, :] = -1e9
                            logits[i, self.na_num] = 1e9

        metrics = self.eval_fn(logits, labels)
        self.log("Test/f1", metrics["f1"])

    def on_save_checkpoint(self, checkpoint):
        """Persist class-adaptive thresholds into the checkpoint."""
        if hasattr(self, "class_thresholds"):
            checkpoint["class_thresholds"] = self.class_thresholds

    def on_load_checkpoint(self, checkpoint):
        """Restore class-adaptive thresholds from the checkpoint."""
        if "class_thresholds" in checkpoint:
            self.class_thresholds = checkpoint["class_thresholds"]

    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="Regularization lambda weight")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="Regularization gamma weight")
        parser.add_argument("--alpha_other", type=float, default=1.0, help="Weight assigned to NA/Other class")
        parser.add_argument(
            "--reject_lambda",
            type=float,
            default=1.0,
            help="Adaptive rejection threshold multiplier (mu - lambda * sigma)",
        )
        parser.add_argument(
            "--use_adaptive_threshold",
            type=int,
            default=1,
            help="Flag to enable class-adaptive threshold rejection (1: enabled, 0: disabled)",
        )
        parser.add_argument(
            "--multi_task_lesson_on",
            type=int,
            default=1,
            help="Flag to enable multi-task curriculum learning (1: enabled, 0: disabled)",
        )
        parser.add_argument(
            "--use_focal_loss",
            type=int,
            default=0,
            help="Flag to use Focal Loss instead of CrossEntropy for relation classification",
        )
        parser.add_argument(
            "--focal_gamma",
            type=float,
            default=1.0,
            help="Focusing parameter gamma for Focal Loss",
        )
        parser.add_argument(
            "--use_task4_veto",
            type=int,
            default=1,
            help="Flag to enable Task 4 veto mechanism (1: enabled, 0: disabled)",
        )
        parser.add_argument(
            "--task4_veto_threshold",
            type=float,
            default=0.9,
            help="Probability threshold for Task 4 veto",
        )
        parser.add_argument(
            "--class_learning_threshold",
            type=float,
            default=0.6,
            help="Upper bound ratio for curriculum learning auxiliary loss",
        )
        return parser

    def compute_multi_mask(self, mask_output, id=0):
        word2label_fir = [i[id][0] for i in self.word2label]
        final_output = torch.zeros_like(mask_output[:, word2label_fir])
        for i in range(len(self.word2label)):
            continue_mul_list = []
            for j in range(len(self.word2label[i][id])):
                continue_mul_list.append(mask_output[:, self.word2label[i][id][j]])
            final_output[:, i] = reduce(lambda x, y: x + y, continue_mul_list)
        final_output = torch.softmax(final_output, dim=-1)
        return final_output

    def pvp_one(self, logits, input_ids, labels=None, hidden_state=None, attention_score=None):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        multi_viewer_final_output = [
            self.compute_multi_mask(mask_output, id=j).view(bs, 1, -1)
            for j in range(self.args.multi_viewer_num)
        ]
        return multi_viewer_final_output, None

    def pvp_multi_viewer(self, logits, input_ids, labels=None, hidden_state=None, attention_score=None):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)

        if self.args.MVRE:
            multi_viewer_mask_idx = [
                mask_idx[[i for i in range(j, len(mask_idx), self.args.multi_viewer_num)]]
                for j in range(self.args.multi_viewer_num)
            ]
            bs = input_ids.shape[0]
            multi_viewer_mask_output = [
                logits[torch.arange(bs), multi_viewer_mask_idx[j]]
                for j in range(self.args.multi_viewer_num)
            ]
            multi_viewer_final_output = [
                self.compute_multi_mask(multi_viewer_mask_output[j], id=j)
                for j in range(self.args.multi_viewer_num)
            ]

            if hidden_state is not None:
                multi_viewer_hidden_state = [
                    hidden_state[torch.arange(bs), multi_viewer_mask_idx[j]]
                    for j in range(self.args.multi_viewer_num)
                ]
                multi_viewer_w = [
                    nn.Sigmoid()(self.predict[j](multi_viewer_hidden_state[j])).view(-1)
                    for j in range(self.args.multi_viewer_num)
                ]

                final_output_list = []
                for i in range(self.args.multi_viewer_num):
                    final_output_list.append(
                        torch.mm(torch.diag_embed(multi_viewer_w[i]), multi_viewer_final_output[i])
                    )
                final_output = reduce(lambda x, y: x + y, final_output_list)
            else:
                final_output = reduce(lambda x, y: x + y, multi_viewer_final_output)

            return final_output, None
        else:
            bs = input_ids.shape[0]
            mask_output = logits[torch.arange(bs), mask_idx]
            assert mask_idx.shape[0] == bs, "only one mask in sequence!"
            final_output = mask_output[:, self.word2label]

        return final_output, None

    def pvp(self, logits, input_ids, labels=None, hidden_state=None, attention_score=None):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:, self.word2label]
        return final_output, None

    def contrastive_loss(self, beta=0.1):
        class_label_emb_all = [[] for _ in range(self.args.multi_viewer_num)]
        like_loss = 0
        unlike_loss = 0
        label_num = len(self.word2label)

        for i in self.word2label:
            class_label_emb = []
            for j in range(self.args.multi_viewer_num):
                class_label_emb.append(self.model.get_output_embeddings().weight[i[j][0]])
                class_label_emb_all[j].append(class_label_emb[-1].unsqueeze(0))

            for j in range(self.args.multi_viewer_num):
                for k in range(j + 1, self.args.multi_viewer_num):
                    like_loss += 1 - torch.nn.functional.cosine_similarity(
                        class_label_emb[j].reshape(1, -1),
                        class_label_emb[k].reshape(1, -1),
                    )

        for j in range(self.args.multi_viewer_num):
            class_label_emb_all[j] = torch.cat(class_label_emb_all[j], dim=0)
            similarity_loss = 1 - torch.cosine_similarity(
                class_label_emb_all[j].unsqueeze(1),
                class_label_emb_all[j].unsqueeze(0),
                dim=-1,
            )
            unlike_loss += similarity_loss.sum()

        if self.args.multi_viewer_num == 1:
            return -beta * unlike_loss / self.args.multi_viewer_num / label_num / (label_num + 1) / 2
        else:
            return (
                like_loss / label_num / self.args.multi_viewer_num / (self.args.multi_viewer_num - 1) / 2
                - beta * unlike_loss / self.args.multi_viewer_num / label_num / (label_num + 1) / 2
            )

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps:
            parameters = self.model.named_parameters()
        else:
            parameters = [next(self.model.named_parameters())]
            parameters.append(("log_vars", self.log_vars))

        optimizer_group_parameters = [
            {
                "params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)],
                "weight_decay": 0,
            },
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.one_epoch_step * 1,
            num_training_steps=self.num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }