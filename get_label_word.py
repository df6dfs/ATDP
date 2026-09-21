import pdb

from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import torch
import json
import argparse

# model_name_or_path = "roberta_large"
# dataset_name = "semeval"
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
args = parser.parse_args()
model_name_or_path = args.model_name_or_path #"roberta-large"
dataset_name = args.dataset_name #"retacred"


def get_temps(tokenizer):
    temps = {}
    with open(f"dataset/{dataset_name}/temp.txt", "r") as f:
        for i in f.readlines():
            i = i.strip().split("\t")
           
            info = {}
            # i[1]是temps里每一行的的第二列 是关系标签
            #   0	Other	nothing	has	nothing	to	nothing
            # # 0	Member-Collection(e1,e2)	member	member	of	collection	collection
            # # 0	Entity-Origin(e1,e2)	entity	entity	of	origin	origin
            # # 0	Cause-Effect(e1,e2)	cause	cause	of	effect	effect
            # # 0	Component-Whole(e1,e2)	component	component	of	whole	whole
            # # 0	Product-Producer(e1,e2)	product	product	of	producer	producer
            # # 0	Instrument-Agency(e1,e2)	instrument	instrument	of	agency	agency
            # # 0	Entity-Destination(e1,e2)	entity	entity	of	destination	destination
            # # 0	Content-Container(e1,e2)	content	content	of	container	container
            # # 0	Message-Topic(e1,e2)	message	message	of	topic	topic
            # # 2	Cause-Effect(e2,e1)	effect	effect	of	cause	cause
            # # 2	Product-Producer(e2,e1)	producer	producer	of	product	product
            # # 2	Component-Whole(e2,e1)	whole	whole	of	component	component
            # # 2	Instrument-Agency(e2,e1)	agency	agency	of	instrument	instrument
            # # 2	Member-Collection(e2,e1)	collection	collection	of	member	member
            # # 2	Message-Topic(e2,e1)	topic	topic	of	message	message
            # # 2	Entity-Origin(e2,e1)	origin	origin	of	entity	entity
            # # 2	Content-Container(e2,e1)	container	container	of	content	content
            # # 2	Entity-Destination(e2,e1)	destination	destination	of	entity	entity
            info['name'] = i[1].strip() 
            info['temp'] = [
                    ['the', tokenizer.mask_token],
                    [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token], 
                    ['the', tokenizer.mask_token],
             ]
            print (i)
            
            info['labels'] = [
                (i[2],),
                (i[3],i[4],i[5]),
                (i[6],)
            ]
            info['label str'] = f"{i[3]} {i[4]} {i[5]}"
            temps[info['name']] = info
           
    
    return temps

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# print(len(tokenizer)) #50265
temps = get_temps(tokenizer)
#print(temps)
    #     temps的结构由name、temp、labels和label str组成 但是这个变量没有被引用
    # 'name' =
    # 'Member-Collection(e1,e2)'
    # 'temp' =
    # [['the', '<mask>'], ['<mask>', '<mask>', '<mask>'], ['the', '<mask>']]
    # 'labels' =
    # [('member',), ('member', 'of', 'collection'), ('collection',)]
    # 'label str' =
    # 'member of collection'
def split_label_words_same_word(tokenizer, label_list,word_list):
    label_word_list = []
    print(len(tokenizer))
    for label,word in  zip(label_list,word_list):

        if label == 'no_relation' or label == "NA" or label == "Other":
            label_words = ['no relation ', 'irrelevant ', 'other ',"miscellaneous ", "unrelated ", "generic ", "neutral ", "nonspecific ",
                           "not applicable ", "unknown ", "non-specified "]
            label_words=[word.lower().replace(":"," ").replace("_"," ").replace("-"," ") for word in label_words ]
            label_word_id = [i  for word in label_words for i in tokenizer.encode(word, add_special_tokens=False)]

            label_word_list.append(torch.tensor(label_word_id))

            print(label, label_word_id)
        else:
            #label = temps[label]["label str"]
            similar_word_list=[]
            for token in word:
               token = token.lower()
               token= token.replace(":"," ").replace("_"," ").replace("per","person").replace("org","organization").replace("-"," ")
               token=token+" "
               similar_word_list.append(token)



            label = label.lower()
            #     去除标签中括号及其后内容：
            #     label.split("(")：将字符串按 ( 分割，生成一个列表。
            #     [0]：取分割后的第一个元素，即 ( 前面的内容。
            label = label.split("(")[0]
            label = label.replace(":"," ").replace("_"," ").replace("per","person").replace("org","organization").replace("-"," ")
            similar_word_list.append(label)
            # 通过分词器给定label的关系词来计算出一个ID，这个id可能是个列表
            #     id其实这个张量
            #     component-whole [46362, 12, 11613, 4104]
            #     instrument-agency [179, 41392, 12, 26904]
            #     member-collection [8648, 12, 44443]
            label_word_id=[i  for word in similar_word_list for i in tokenizer.encode(word, add_special_tokens=False)]


            label_word_list.append(torch.tensor(label_word_id))
            print(label, label_word_id)
            # 用于对张量列表进行填充，使所有张量的长度相同 用0填充短的张量

    padded_label_word_list = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)
    return padded_label_word_list
def split_label_words(tokenizer, label_list):
    label_word_list = []
    print(len(tokenizer))

    for label in label_list:
        if label == 'no_relation' or label == "NA":
            label_word_id = tokenizer.encode('no relation', add_special_tokens=False)
            # label_words = ['no relation ', 'irrelevant ', 'other ', "miscellaneous ", "unrelated ", "generic ",
            #                "neutral ", "nonspecific ",
            #                "not applicable ", "unknown ", "non-specified "]
            # label_words = [word.lower().replace(":", " ").replace("_", " ") for word in label_words]
            # label_word_id = [i for word in label_words for i in tokenizer.encode(word, add_special_tokens=False)]

            label_word_list.append(torch.tensor(label_word_id))

            print(label, label_word_id)

        else:
            #label = temps[label]["label str"]
            label = label.lower()
            label = label.split("(")[0]
            label = label.replace(":"," ").replace("_"," ").replace("per","person").replace("org","organization")
            label_word_id = tokenizer(label, add_special_tokens=False)['input_ids']
            print(label, label_word_id)
            label_word_list.append(torch.tensor(label_word_id))
    padded_label_word_list = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)
    return padded_label_word_list

label_list = []
with open(f"dataset/{dataset_name}/rel2id.json", "r") as file:
    t = json.load(file)
    id_dict = {}
    max_v = 0
    for k, v in t.items():
        id_dict[str(v)] = k
        max_v = max(max_v, v)
    for i in range(max_v + 1):
        label_list.append(id_dict[str(i)])
print('aaa')
print(label_list)

# word_list=[]
# with open(f"./dataset/{dataset_name}/diction_{dataset_name}.json", 'r', encoding='utf-8') as f:
#     loaded_data = json.load(f)
#     word_dict= {}
#     max_id=0
#     for relation_type, details in loaded_data.items():
#         max_id=max(max_id,details['id'])
#         word_dict[str(details['id'])]=details['words']
#     for i in range(max_id+1):
#         word_list.append(word_dict[str(i)])

# print(word_list)



#   label_list是一个列表 ，值如下
   
# 00 =
# 'Other'
# 01 =
# 'Component-Whole(e2,e1)'
# 02 =
# 'Instrument-Agency(e2,e1)'
# 03 =
# 'Member-Collection(e1,e2)'
# 04 =
# 'Cause-Effect(e2,e1)'
# 05 =
# 'Entity-Destination(e1,e2)'
# 06 =
# 'Content-Container(e1,e2)'
# 07 =
# 'Message-Topic(e1,e2)'
# 08 =
# 'Product-Producer(e2,e1)'
# 09 =
# 'Member-Collection(e2,e1)'
# 10 =
# 'Entity-Origin(e1,e2)'
# 11 =
# 'Cause-Effect(e1,e2)'
# 12 =
# 'Component-Whole(e1,e2)'
# 13 =
# 'Message-Topic(e2,e1)'
# 14 =
# 'Product-Producer(e1,e2)'
# 15 =
# 'Entity-Origin(e2,e1)'
# 16 =
# 'Content-Container(e2,e1)'
# 17 =
# 'Instrument-Agency(e1,e2)'
# 18 =
# 'Entity-Destination(e2,e1)'
# ------------------------

# t = split_label_words_same_word(tokenizer, label_list,word_list)
t = split_label_words(tokenizer, label_list)
print(t)
print(t.shape)
# 这个pt文件存放着每个关系的一个被用0填充后的张量列表

model_name_or_path="roberta-large"
with open(f"./dataset/{model_name_or_path}_{dataset_name}.pt", "wb") as file:
    torch.save(t, file)