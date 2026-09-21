import pdb
from typing import Any, Optional, Union
from .base_data_module import BaseDataModule
from dataclasses import dataclass
from .processor import get_dataset, processors
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
import shutil
import os


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


class DIALOGUE(BaseDataModule):
    def __init__(self, args, model) -> None:
        super().__init__(args)
        self.processor = processors[self.args.task_name](self.args.data_dir, self.args.use_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

        self.num_labels = len(self.processor.get_labels())

        class_list = [f"[class{i}]" for i in range(1, self.num_labels + 1)]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': class_list})
        unused_list = [f"[unused{i}]" for i in range(1, 50)]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': unused_list})
        speaker_list = [f"[speaker{i}]" for i in range(1, 50)]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': speaker_list})
        so_list = ["[sub]", "[obj]"]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': so_list})

    def setup(self, stage=None):
        self.data_train = get_dataset("train", self.args, self.tokenizer, self.processor)
        self.data_val = get_dataset("dev", self.args, self.tokenizer, self.processor)
        self.data_test = get_dataset("test", self.args, self.tokenizer, self.processor)

    def prepare_data(self):
        pass

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--task_name", type=str, default="normal", help="[normal, reloss, ptune]")
        parser.add_argument("--model_name_or_path", type=str, default="/home/xx/bert-base-uncased",
                            help="Number of examples to operate on per forward step.")
        parser.add_argument("--max_seq_length", type=int, default=512,
                            help="Number of examples to operate on per forward step.")
        parser.add_argument("--ptune_k", type=int, default=7, help="number of unused tokens in prompt")
        return parser


class WIKI80(BaseDataModule):
    def __init__(self, args, model=None) -> None:
        super().__init__(args)
        self.processor = processors[self.args.task_name](self.args.data_dir, self.args.use_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

        use_gpt = "gpt" in args.model_name_or_path

        rel2id = self.processor.get_labels()
        self.num_labels = len(rel2id)

        entity_list = ["[object_start]", "[object_end]", "[subject_start]", "[subject_end]", "[e1]", "[/e1]", "[e2]",
                       "[/e2]"]
        class_list = [f"[class{i}]" for i in range(1, self.num_labels + 1)]
        # 增加多视角的关系类别特殊标识符
        MVRE_class_list = [f"[class{i}_{j}]" for i in range(1, self.num_labels + 1) for j in
                           range(args.multi_viewer_num)]
        rel_split_lits = ["[REL]", "[REL_SEP]"]
        contex_sep_list = ["[SEP]"]

        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_list})
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': class_list})

        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': MVRE_class_list})
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': rel_split_lits})
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': contex_sep_list})

        if use_gpt:
            self.tokenizer.add_special_tokens({'cls_token': "[CLS]"})
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        so_list = ["[sub]", "[obj]"]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': so_list})

        prompt_tokens = [f"[T{i}]" for i in range(1, 6)]
        self.tokenizer.add_special_tokens({'additional_special_tokens': prompt_tokens})

        for a in range(1, self.num_labels + 1):
            if args.MVRE:
                for j in range(args.multi_viewer_num):
                    self.tokenizer.add_tokens(f"[class{a}_{j}]", special_tokens=True)

    def setup(self, stage=None):
        self.data_train = get_dataset("train", self.args, self.tokenizer, self.processor)
        self.data_val = get_dataset("dev", self.args, self.tokenizer, self.processor)
        self.data_test = get_dataset("test", self.args, self.tokenizer, self.processor)

    def prepare_data(self):
        pass

    def get_tokenizer(self):
        return self.tokenizer

    def get_custom_dataloader(self, batch_size=None):
        """
        专门用于加载指定文件进行预测的 DataLoader
        """
        if batch_size is None:
            batch_size = self.batch_size  # 或者 self.args.batch_size

        # 这里复用你原本用于构建 dataset 的逻辑
        # 假设你有一个 Dataset 类叫 MyDataset (就是 self.train_dataset 用的那个类)
        # 注意：你需要确保你的 Dataset 类能处理没有标签的数据，或者在 file_path 文件里预先填入 dummy label

        # 这里的 MyDataset 是你当前文件里使用的 Dataset 类名
        # processor 是你处理数据的逻辑，确保 mode='test' 或其它不依赖真实标签的模式
        dataset = get_dataset('predict', self.args, self.tokenizer, self.processor)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # 预测时不要打乱，方便后续按顺序合并
            num_workers=4,
            pin_memory=True
        )

        return dataloader

    @staticmethod
    def add_to_argparse(parser):

        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--task_name", type=str, default="normal", help="[normal, reloss, ptune]")
        parser.add_argument("--model_name_or_path", type=str, default="/home/xx/bert-base-uncased",
                            help="Number of examples to operate on per forward step.")
        parser.add_argument("--max_seq_length", type=int, default=512,
                            help="Number of examples to operate on per forward step.")
        parser.add_argument("--ptune_k", type=int, default=7, help="number of unused tokens in prompt")
        return parser