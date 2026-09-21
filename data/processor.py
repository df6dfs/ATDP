import csv
import pdb
import pickle
import os
import json
import logging
import torch
from torch.utils.data import TensorDataset, Dataset
from collections import OrderedDict
import re
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

keyword_files = ["keyword_train.txt", "keyword_dev.txt", "keyword_test.txt"]
semeval_definitions = {
    "Cause-Effect(e1,e2)": "Entity 1 is the cause that leads to the effect of Entity 2.",
    "Cause-Effect(e2,e1)": "Entity 2 is the cause that leads to the effect of Entity 1.",

    "Instrument-Agency(e1,e2)": "Entity 1 is the instrument or tool used by the agent Entity 2.",
    "Instrument-Agency(e2,e1)": "Entity 2 is the instrument or tool used by the agent Entity 1.",

    "Product-Producer(e1,e2)": "Entity 1 is the product created, produced, or manufactured by Entity 2.",
    "Product-Producer(e2,e1)": "Entity 2 is the product created, produced, or manufactured by Entity 1.",

    "Content-Container(e1,e2)": "Entity 1 is the content physically stored or contained within Entity 2.",
    "Content-Container(e2,e1)": "Entity 2 is the content physically stored or contained within Entity 1.",

    "Entity-Origin(e1,e2)": "Entity 1 is an entity that comes from or originates from the source Entity 2.",
    "Entity-Origin(e2,e1)": "Entity 2 is an entity that comes from or originates from the source Entity 1.",

    "Entity-Destination(e1,e2)": "Entity 1 is an entity that is moving towards the destination Entity 2.",
    "Entity-Destination(e2,e1)": "Entity 2 is an entity that is moving towards the destination Entity 1.",

    "Component-Whole(e1,e2)": "Entity 1 is a component or a constituent part that makes up the whole Entity 2.",
    "Component-Whole(e2,e1)": "Entity 2 is a component or a constituent part that makes up the whole Entity 1.",

    "Member-Collection(e1,e2)": "Entity 1 is a member that forms a non-functional part of the collection Entity 2.",
    "Member-Collection(e2,e1)": "Entity 2 is a member that forms a non-functional part of the collection Entity 1.",

    "Message-Topic(e1,e2)": "Entity 1 is a written or spoken message communicating information about the topic Entity 2.",
    "Message-Topic(e2,e1)": "Entity 2 is a written or spoken message communicating information about the topic Entity 1.",

    "Other": "The relationship between Entity 1 and Entity 2 does not belong to any of the predefined specific categories."
}

tacred_definitions = {
    # ================= Organization Relations (16 types) =================
    "org:alternate_names": "Any alternative name, alias, or acronym used by the subject organization.",
    "org:city_of_headquarters": "The city where the headquarters of the subject organization is located.",
    "org:country_of_headquarters": "The country where the headquarters of the subject organization is located.",
    "org:dissolved": "The date when the subject organization was dissolved or ceased to exist.",
    "org:founded": "The date when the subject organization was founded or established.",
    "org:founded_by": "The person or organization that founded or established the subject organization.",
    "org:member_of": "The larger organization or group that the subject organization is a member of.",
    "org:members": "Other organizations or people that are members of the subject organization.",
    "org:number_of_employees/members": "The total number of employees or members belonging to the subject organization.",
    "org:parents": "The parent organization or holding company that owns the subject organization.",
    "org:political/religious_affiliation": "The political or religious ideology or group affiliated with the subject organization.",
    "org:shareholders": "The person or organization that holds shares or invests in the subject organization.",
    "org:stateorprovince_of_headquarters": "The state or province where the headquarters of the subject organization is located.",
    "org:subsidiaries": "The subsidiary organization owned or controlled by the subject parent organization.",
    "org:top_members/employees": "The top-level members, executives, or employees of the subject organization.",
    "org:website": "The official URL or website of the subject organization.",

    # ================= Person Relations (25 types) =================
    "per:age": "The age of the subject person.",
    "per:alternate_names": "Alternative names, aliases, or nicknames of the subject person.",
    "per:cause_of_death": "The explicitly stated cause of death of the subject person.",
    "per:charges": "The criminal charges, accusations, or convictions brought against the subject person.",
    "per:children": "The children of the subject person.",
    "per:cities_of_residence": "The city where the subject person lives or has lived.",
    "per:city_of_birth": "The city where the subject person was born.",
    "per:city_of_death": "The city where the subject person died.",
    "per:countries_of_residence": "The country where the subject person lives or has lived.",
    "per:country_of_birth": "The country where the subject person was born.",
    "per:country_of_death": "The country where the subject person died.",
    "per:date_of_birth": "The date when the subject person was born.",
    "per:date_of_death": "The date when the subject person died.",
    "per:employee_of": "The organization or company that the subject person is employed by or works for.",
    "per:origin": "The nationality, ethnicity, or country of origin of the subject person.",
    "per:other_family": "Family members of the subject person other than parents, children, siblings, or spouse.",
    "per:parents": "The parents of the subject person.",
    "per:religion": "The religion or religious affiliation of the subject person.",
    "per:schools_attended": "The school, college, or university attended by the subject person.",
    "per:siblings": "The brothers or sisters of the subject person.",
    "per:spouse": "The husband, wife, or married partner of the subject person.",
    "per:stateorprovince_of_birth": "The state or province where the subject person was born.",
    "per:stateorprovince_of_death": "The state or province where the subject person died.",
    "per:stateorprovinces_of_residence": "The state or province where the subject person lives or has lived.",
    "per:title": "The official job title, profession, or position of the subject person.",

    # ================= No Relation (1 type) =================
    "NA": "There is no predefined specific relationship between the subject entity and the object entity."
}

tacrev_definitions = {
    # ================= Organization Relations (16 types) =================
    "org:alternate_names": "Any alternative name, alias, or acronym used by the subject organization.",
    "org:city_of_headquarters": "The city where the headquarters of the subject organization is located.",
    "org:country_of_headquarters": "The country where the headquarters of the subject organization is located.",
    "org:dissolved": "The date when the subject organization was dissolved or ceased to exist.",
    "org:founded": "The date when the subject organization was founded or established.",
    "org:founded_by": "The person or organization that founded or established the subject organization.",
    "org:member_of": "The larger organization or group that the subject organization is a member of.",
    "org:members": "Other organizations or people that are members of the subject organization.",
    "org:number_of_employees/members": "The total number of employees or members belonging to the subject organization.",
    "org:parents": "The parent organization or holding company that owns the subject organization.",
    "org:political/religious_affiliation": "The political or religious ideology or group affiliated with the subject organization.",
    "org:shareholders": "The person or organization that holds shares or invests in the subject organization.",
    "org:stateorprovince_of_headquarters": "The state or province where the headquarters of the subject organization is located.",
    "org:subsidiaries": "The subsidiary organization owned or controlled by the subject parent organization.",
    "org:top_members/employees": "The top-level members, executives, or employees of the subject organization.",
    "org:website": "The official URL or website of the subject organization.",

    # ================= Person Relations (25 types) =================
    "per:age": "The age of the subject person.",
    "per:alternate_names": "Alternative names, aliases, or nicknames of the subject person.",
    "per:cause_of_death": "The explicitly stated cause of death of the subject person.",
    "per:charges": "The criminal charges, accusations, or convictions brought against the subject person.",
    "per:children": "The children of the subject person.",
    "per:cities_of_residence": "The city where the subject person lives or has lived.",
    "per:city_of_birth": "The city where the subject person was born.",
    "per:city_of_death": "The city where the subject person died.",
    "per:countries_of_residence": "The country where the subject person lives or has lived.",
    "per:country_of_birth": "The country where the subject person was born.",
    "per:country_of_death": "The country where the subject person died.",
    "per:date_of_birth": "The date when the subject person was born.",
    "per:date_of_death": "The date when the subject person died.",
    "per:employee_of": "The organization or company that the subject person is employed by or works for.",
    "per:origin": "The nationality, ethnicity, or country of origin of the subject person.",
    "per:other_family": "Family members of the subject person other than parents, children, siblings, or spouse.",
    "per:parents": "The parents of the subject person.",
    "per:religion": "The religion or religious affiliation of the subject person.",
    "per:schools_attended": "The school, college, or university attended by the subject person.",
    "per:siblings": "The brothers or sisters of the subject person.",
    "per:spouse": "The husband, wife, or married partner of the subject person.",
    "per:stateorprovince_of_birth": "The state or province where the subject person was born.",
    "per:stateorprovince_of_death": "The state or province where the subject person died.",
    "per:stateorprovinces_of_residence": "The state or province where the subject person lives or has lived.",
    "per:title": "The official job title, profession, or position of the subject person.",

    # ================= No Relation (1 type) =================
    "no_relation": "There is no predefined specific relationship between the subject entity and the object entity."
}


def tokenize(text, tokenizer):
    # berts tokenize ways
    # tokenize the [unused12345678910]
    D = [f"[unused{i}]" for i in range(10)]
    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t) - 1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    for t in textraw:
        if t in D:
            text += [t]
        else:
            tokens = tokenizer.tokenize(t, add_special_tokens=False)
            for tok in tokens:
                text += [tok]

    for idx, t in enumerate(text):
        if idx + 3 < len(text) and t == "[" and text[idx + 1] == "[UNK]" and text[idx + 2] == "]":
            text = text[:idx] + ["[MASK]"] + text[idx + 3:]

    return text


def clean_text(text_o):
    isstr = 0
    if isinstance(text_o, str):
        text_o = text_o.split(' ')
        isstr = 1
    text_new = []
    for text in text_o:
        pattern_url1 = re.compile(
            r'(https?|ftp|file|img3):\/\/[a-z0-9_.:]+\/[-a-z0-9_:@&?=+,.!/~*%$]*(\.(html|htm|shtml))?')
        pattern_url2 = re.compile(r'^https?:\/\/([^/:]+)(:(\d)+)?(/.*)?$')
        pattern_url6 = re.compile(r'(www.)[a-zA-Z0-9\-\.]+')
        text_url1 = re.sub(pattern=pattern_url1, repl='', string=str(text))
        text_url2 = re.sub(pattern=pattern_url2, repl='', string=str(text_url1))
        text = re.sub(pattern=pattern_url6, repl='', string=str(text_url2))
        text_new.append(text)
    if isstr:
        text_out = " ".join(text_new)
    text_out = text_new
    return text_out


n_class = 1


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None, entity=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.entity = entity


class InputExampleWiki80(object):
    """A single training/test example for span pair classification."""

    def __init__(self, guid, sentence, span1, span2, ner1, ner2, label):
        self.guid = guid
        self.sentence = sentence
        self.span1 = span1
        self.span2 = span2
        self.ner1 = ner1
        self.ner2 = ner2
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, entity=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entity = entity


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class bertProcessor(DataProcessor):  # bert_s
    def __init__(self, data_path="data", use_prompt=False):
        def is_speaker(a):
            a = a.split()
            return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()

        # replace the speaker with [unused] token
        def rename(d, x, y):
            d = d.replace("’", "'")
            d = d.replace("im", "i")
            d = d.replace("...", ".")
            unused = ["[unused1]", "[unused2]"]
            a = []
            if is_speaker(x):
                a += [x]
            else:
                a += [None]
            if x != y and is_speaker(y):
                a += [y]
            else:
                a += [None]
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", unused[i] + " :")
                if x == a[i]:
                    x = unused[i]
                if y == a[i]:
                    y = unused[i]
            return d, x, y

        self.D = [[], [], []]
        for sid in range(3):
            # Split into three subsets: train, dev, and test
            with open(data_path + "/" + ["train.json", "dev.json", "test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            sample_idx = 0
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    for k in range(36):
                        if k + 1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    d, h, t = rename(' '.join(data[i][0]).lower(), data[i][1][j]["x"].lower(),
                                     data[i][1][j]["y"].lower())
                    if use_prompt:
                        prompt = f"{h} is the <mask> {t} ."
                    else:
                        prompt = f"what is the relation between {h} and {t} ?"
                    sample_idx += 1
                    d = [
                        prompt + d,
                        h,
                        t,
                        rid,
                    ]
                    self.D[sid] += [d]
        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(36)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=clean_text(data[i][0]), text_b=data[i][1], label=data[i][3],
                                         text_c=data[i][2]))
        return examples


class wiki80Processor(DataProcessor):
    """Processor for the TACRED data set."""

    def __init__(self, data_path, use_prompt):
        super().__init__()
        self.data_dir = data_path

    @classmethod
    def _read_json(cls, input_file):
        data = []
        with open(input_file, "r", encoding='utf-8') as reader:
            all_lines = reader.readlines()
            for line in all_lines:
                ins = eval(line)
                data.append(ins)
        return data

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "val.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.txt")), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "llama3_augmented_train.txt")), "predict")

    def get_labels(self, negative_label="no_relation"):
        data_dir = self.data_dir
        """See base class."""
        with open(os.path.join(data_dir, 'rel2id.json'), "r", encoding='utf-8') as reader:
            re2id = json.load(reader)
        return re2id

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in dataset:
            sentence = example['token']
            sentence = clean_text(sentence)
            examples.append(InputExampleWiki80(guid=None,
                                               sentence=sentence,
                                               # maybe some bugs here, I don't -1
                                               span1=(example['h']['pos'][0], example['h']['pos'][1]),
                                               span2=(example['t']['pos'][0], example['t']['pos'][1]),
                                               ner1=None,
                                               ner2=None,
                                               label=example['relation']))
        return examples


def convert_examples_to_features_normal(examples, max_seq_length, tokenizer):
    print("#examples", len(examples))
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenize(example.text_a, tokenizer)
        tokens_b = tokenize(example.text_b, tokenizer)
        tokens_c = tokenize(example.text_c, tokenizer)

        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_b + ["[SEP]"] + tokens_c

        inputs = tokenizer(
            example.text_a,
            example.text_b + tokenizer.sep_token + example.text_c,
            truncation="longest_first",
            max_length=max_seq_length,
            padding="max_length",
            add_special_tokens=True
        )

        label_id = example.label

        if ex_index == 0:
            logger.info(f"input_text : {tokens_a} {tokens_b} {tokens_c}")
            logger.info(f"input_ids : {inputs['input_ids']}")

        # append 1 sample with 2 input
        features.append(
            InputFeatures(
                input_ids=inputs['input_ids'],
                input_mask=inputs['attention_mask'],
                segment_ids=inputs['attention_mask'],
                label_id=label_id,
            )
        )

    print('#features', len(features))
    return features


# ======= Mode Parameter Acceptance =======
def convert_examples_to_features(examples, max_seq_length, tokenizer, args, rel2id, mode="train"):
    """Loads a data file into a list of `InputBatch`s."""

    save_file = "./dataset/cached_wiki80.pkl"
    # Renamed to text_mode to avoid collision with incoming dataset mode (train/dev/test)
    text_mode = "text"

    num_tokens = 0
    num_fit_examples = 0
    instances = []

    use_bert = "BertTokenizer" in tokenizer.__class__.__name__
    use_gpt = "GPT" in tokenizer.__class__.__name__

    assert not (use_bert and use_gpt), "model cannot be gpt and bert together"
    # Pre-fetch special marker IDs to avoid redundant token lookups inside the loop
    sub_marker_id = tokenizer.convert_tokens_to_ids("[sub]")
    obj_marker_id = tokenizer.convert_tokens_to_ids("[obj]")
    mask_token_id = tokenizer.mask_token_id

    print('loading..')
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # === 1. Construct base context tokens ===
        tokens = []
        SUBJECT_START = "[subject_start]"
        SUBJECT_END = "[subject_end]"
        OBJECT_START = "[object_start]"
        OBJECT_END = "[object_end]"

        if text_mode.startswith("text"):
            for i, token in enumerate(example.sentence):
                if i == example.span1[0]: tokens.append(SUBJECT_START)
                if i == example.span1[1]: tokens.append(SUBJECT_END)
                if i == example.span2[0]: tokens.append(OBJECT_START)
                if i == example.span2[1]: tokens.append(OBJECT_END)
                tokens.append(token)

        SUBJECT = " ".join(example.sentence[example.span1[0]: example.span1[1]])
        OBJECT = " ".join(example.sentence[example.span2[0]: example.span2[1]])

        # Extract current ground-truth class and generate corresponding virtual relation tokens (e.g., [class1_0] [class1_1] [class1_2])
        current_label_id = rel2id[example.label]
        class_i = list(rel2id.values()).index(current_label_id) + 1
        rel_classes_str = " ".join([f"[class{class_i}_{j}]" for j in range(args.multi_viewer_num)])

        # Fully masked relation string (for the main relation extraction task)
        masked_rel_str = " ".join([tokenizer.mask_token] * args.multi_viewer_num)

        rel_part1 = "[sub]"
        rel_part2 = "[obj]"
        context_str = " ".join(tokens)
        soft_prefix = f"{tokenizer.unk_token} " * args.num_soft_tokens if args.soft_prompt_switch == 0 else ""

        # =====================================================================
        # [Task 1: Main Task RE] Mask intermediate virtual relation tokens: [class_i_j] -> [MASK]
        # =====================================================================
        prompt_main = f"{rel_part1} {SUBJECT} {rel_part1} {masked_rel_str} {rel_part2} {OBJECT} {rel_part2} ."
        strs_main = f"{soft_prefix}{context_str} {prompt_main} {context_str}"

        inputs_main = tokenizer(
            strs_main, truncation="longest_first", max_length=max_seq_length,
            padding="max_length", add_special_tokens=True
        )
        main_input_ids = inputs_main['input_ids']
        main_attention_mask = inputs_main['attention_mask']

        # =====================================================================
        # [Task 2 & 3 Foundation] Construct full unmasked sentence with explicit entities and relation tokens for uniform tokenization
        # Current prompt structure: [sub] Subject [sub] [class1_0][class1_1] [obj] Object [obj] .
        # =====================================================================
        prompt_unmasked = f"{rel_part1} {SUBJECT} {rel_part1} {rel_classes_str} {rel_part2} {OBJECT} {rel_part2} ."
        strs_unmasked = f"{soft_prefix}{context_str} {prompt_unmasked} {context_str}"

        inputs_unmasked = tokenizer(
            strs_unmasked, truncation="longest_first", max_length=max_seq_length,
            padding="max_length", add_special_tokens=True
        )
        base_input_ids = inputs_unmasked['input_ids']

        # Precisely locate prompt entity spans at the input_ids level
        sub_positions = [i for i, x in enumerate(base_input_ids) if x == sub_marker_id]
        obj_positions = [i for i, x in enumerate(base_input_ids) if x == obj_marker_id]

        # Ensure delimiter tokens exist within sequence bounds; skip if truncated
        if len(sub_positions) < 2 or len(obj_positions) < 2:
            continue

        sub_st, sub_ed = sub_positions[0] + 1, sub_positions[1]
        obj_st, obj_ed = obj_positions[0] + 1, obj_positions[1]

        # =====================================================================
        # [Task 2: Subject MLM] Mask the subject span in base_input_ids
        # =====================================================================
        task2_input_ids = list(base_input_ids)
        task2_mlm_label = [-100] * max_seq_length

        for i in range(sub_st, sub_ed):
            task2_mlm_label[i] = task2_input_ids[i]  # Record ground-truth token ID as target
            task2_input_ids[i] = mask_token_id       # Replace with MASK ID

        # =====================================================================
        # [Task 3: Object MLM] Mask the object span in base_input_ids
        # =====================================================================
        task3_input_ids = list(base_input_ids)
        task3_mlm_label = [-100] * max_seq_length

        for i in range(obj_st, obj_ed):
            task3_mlm_label[i] = task3_input_ids[i]  # Record ground-truth token ID as target
            task3_input_ids[i] = mask_token_id       # Replace with MASK ID

        # =====================================================================
        # [Task 4: Relation Existence Detection MLM]
        # =====================================================================
        # Determine whether a relation holds between entities (compatible with datasets having NA/Other categories)
        is_related = False if example.label in ['NA', 'no_relation', 'Other'] else True
        # Use 'highly' for positive relation existence and 'not' for non-existence (leading space matches RoBERTa BPE)
        target_task4_id = tokenizer.encode(" highly", add_special_tokens=False)[0] if is_related else \
            tokenizer.encode(" not", add_special_tokens=False)[0]

        # Declarative template for relation verification
        prompt_task4 = f"{rel_part1} {SUBJECT} {rel_part1} and {rel_part2} {OBJECT} {rel_part2} are {tokenizer.mask_token} related ."
        strs_task4 = f"{soft_prefix}{context_str} {prompt_task4} {context_str}"

        inputs_task4 = tokenizer(
            strs_task4, truncation="longest_first", max_length=max_seq_length,
            padding="max_length", add_special_tokens=True
        )
        task4_input_ids = inputs_task4['input_ids']
        task4_attention_mask = inputs_task4['attention_mask']  # Task 4 prompt length differs, requires dedicated mask
        task4_mlm_label = [-100] * max_seq_length

        # Locate [MASK] in prompt and assign classification target
        try:
            task4_mask_idx = task4_input_ids.index(mask_token_id)
            task4_mlm_label[task4_mask_idx] = target_task4_id
        except ValueError:
            pass  # Ignore Task 4 loss computation if [MASK] was truncated

        # === Assemble Instance Data ===
        x = OrderedDict()
        x['main_input_ids'] = main_input_ids
        x['main_attention_mask'] = main_attention_mask

        x['task2_input_ids'] = task2_input_ids
        x['task2_mlm_label'] = task2_mlm_label

        x['task3_input_ids'] = task3_input_ids
        x['task3_mlm_label'] = task3_mlm_label

        x['task4_input_ids'] = task4_input_ids
        x['task4_attention_mask'] = task4_attention_mask
        x['task4_mlm_label'] = task4_mlm_label

        x['label'] = current_label_id
        x['index'] = ex_index

        instances.append(x)

    # ---------------- Convert to Tensors and build Dataset ----------------
    # Convert lists into tensors of shape [dataset_size, max_seq_length]
    t_main_input_ids = torch.tensor([o['main_input_ids'] for o in instances], dtype=torch.long)
    t_main_attention_mask = torch.tensor([o['main_attention_mask'] for o in instances], dtype=torch.long)

    t_task2_input_ids = torch.tensor([o['task2_input_ids'] for o in instances], dtype=torch.long)
    t_task2_mlm_label = torch.tensor([o['task2_mlm_label'] for o in instances], dtype=torch.long)

    t_task3_input_ids = torch.tensor([o['task3_input_ids'] for o in instances], dtype=torch.long)
    t_task3_mlm_label = torch.tensor([o['task3_mlm_label'] for o in instances], dtype=torch.long)

    # Task 4 tensors
    t_task4_input_ids = torch.tensor([o['task4_input_ids'] for o in instances], dtype=torch.long)
    t_task4_attention_mask = torch.tensor([o['task4_attention_mask'] for o in instances], dtype=torch.long)
    t_task4_mlm_label = torch.tensor([o['task4_mlm_label'] for o in instances], dtype=torch.long)

    t_labels = torch.tensor([o['label'] for o in instances], dtype=torch.long)

    logger.info("Generated %d strictly aligned multi-task instances." % len(instances))

    # Pack all tensors into a TensorDataset
    dataset = TensorDataset(
        t_main_input_ids,        # [0] Main Task Input [bs, max_seq_length]
        t_main_attention_mask,   # [1] Main Task Attention Mask
        t_task2_input_ids,       # [2] Subject MLM Input [bs, max_seq_length]
        t_task2_mlm_label,       # [3] Subject MLM Label
        t_task3_input_ids,       # [4] Object MLM Input [bs, max_seq_length]
        t_task3_mlm_label,       # [5] Object MLM Label
        t_task4_input_ids,       # [6] Relation Existence Input
        t_task4_attention_mask,  # [7] Relation Existence Attention Mask
        t_task4_mlm_label,       # [8] Relation Existence MLM Label
        t_labels                 # [9] Relation Classification Label
    )

    return dataset


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def get_dataset(mode, args, tokenizer, processor):
    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    elif mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    elif mode == "predict":
        examples = processor.get_predict_examples(args.data_dir)
    else:
        raise Exception("mode must be in choice [train, dev, test, predict]")
    gpt_mode = "wiki80" in args.task_name

    if "wiki80" in args.task_name and "bart" not in args.model_name_or_path and "t5" not in args.model_name_or_path:
        # Normal relation extraction task

        dataset = convert_examples_to_features(
            examples, args.max_seq_length, tokenizer, args, processor.get_labels(),
            mode=mode
        )
        return dataset
    else:
        train_features = convert_examples_to_features_normal(
            examples, args.max_seq_length, tokenizer
        )

    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []
    entity_id = []

    for f in train_features:
        input_ids.append(f.input_ids)
        input_mask.append(f.input_mask)
        segment_ids.append(f.segment_ids)
        label_id.append(f.label_id)

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(label_id, dtype=torch.float)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return train_data


def collate_fn(batch):
    pass


processors = {"normal": bertProcessor, "wiki80": wiki80Processor}