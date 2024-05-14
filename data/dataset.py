import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torch
import torch.nn.functional as F
# from lavis.models import load_model_and_preprocess
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
import numpy as np
import re
# import cv2
from tqdm import tqdm
import clip
def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
def preprocessing(text):
  input_text = text
  input_text = input_text.lower()

  # Removing periods except if it occurs as decimal
  input_text = re.sub(r'(?<!\d)\.(?!\d)', '', input_text)

  # Converting number words to digits
  number_words = {
      "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
      "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
      "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
      "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
      "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
      "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
      "eighty": "80", "ninety": "90"
  }
  input_text = re.sub(r'\b(?:' + '|'.join(number_words.keys()) + r')\b', lambda x: number_words[x.group()], input_text)

  # Removing articles (a, an, the)
  if len(input_text)>3:
    input_text = re.sub(r'\b(?:a|an|the)\b', '', input_text)

  # Adding apostrophe if a contraction is missing it
  input_text = re.sub(r'\b(\w+(?<!e)(?<!a))nt\b', r"\1n't", input_text)

  # input_text = re.sub(r'\b(\w+(?<!t))ent\b', r"\1en't", input_text)

  # Replacing all punctuation (except apostrophe and colon) withinput_text a space character
  input_text = re.sub(r'[^\w\':]|(?<=\d),(?=\d)', ' ', input_text)

  # Removing extra spaces
  input_text = re.sub(r'\s+', ' ', input_text).strip()

  return input_text

class VQAv2Dataset(Dataset):
    IMAGE_PATH = {
        "train": ("train2014","v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json","train2014"),
        "val": ("val2014","v2_OpenEnded_mscoco_val2014_questions.json", "v2_mscoco_val2014_annotations.json","val2014"),
        "train_ab": ("train2015","OpenEnded_abstract_v002_train2015_questions.json", "abstract_v002_train2015_annotations.json","train2015"),
        "val_ab": ("val2015","OpenEnded_abstract_v002_val2015_questions.json", "abstract_v002_val2015_annotations.json","val2015"),
        "testdev": ("test2015", "v2_OpenEnded_mscoco_test-dev2015_questions.json"),
        "test": ("test2015", "v2_OpenEnded_mscoco_test2015_questions.json"),
        "train_gqa":("images","train_balanced_questions.json"),
        "val_gqa":("images","val_balanced_questions.json"),
        "train_vg":("images","question_answers.json"),
        "val_vg":("images","question_answers.json")}
    def __init__(self,root,split,dataset,transform):
        # split for dataset
        # train,val for VQAv2 dataset
        # train_ab,val_ab for VQA Abs dataset
        # train_gqa,val_gqa for GQA dataset
        # train_vg, val_vg for VG dataset
        self.split = split
        # path to the root folder of the dataset
        self.root = root
        # dataset name : VQAv2, VQA Ab, GQA and VG
        self.dataset = dataset
        self.transform = transform
        self.selection = most_common_from_dict
        path = os.path.expanduser(os.path.join(root, self.IMAGE_PATH[split][1]))

        with open(path, 'r') as f:
            data = json.load(f)
        if dataset=="VQAv2" or dataset=="VQAab":
            df = pd.DataFrame(data["questions"])
            if dataset=="VQAv2":
                df["image_path"] = df["image_id"].apply(
                        lambda x: f"{self.IMAGE_PATH[split][0]}/COCO_{self.IMAGE_PATH[split][3]}_{x:012d}.jpg")
            elif dataset=="VQAab":
                    df["image_path"] = df["image_id"].apply(
                        lambda x: f"{self.IMAGE_PATH[split][0]}/abstract_v002_{self.IMAGE_PATH[split][3]}_{x:012d}.png")
            path = os.path.expanduser(os.path.join(root, self.IMAGE_PATH[split][2]))
            with open(path, 'r') as f:
                        data = json.load(f)
            df_annotations = pd.DataFrame(data["annotations"])
            i=0
            vocab_path = '/raid/biplab/hassan/VQA_CLIP/vqa_common_ab.txt'
            with open(vocab_path, 'r') as file:
                for line in file:
                    line = line.replace('\n','')
                    self.vocab[line]=i
                    i+=1
            indices=[]
            for i in range(len(df_annotations)):
                    selected_answer = self.selection(df_annotations["answers"][i])
                    if selected_answer not in self.vocab.keys():
                        indices.append(i)
            df_annotations.drop(indices,axis=0,inplace=True)
            df_annotations.reset_index(inplace=True,drop=True) 
            #    print(df_annotations)
            df = pd.merge(df, df_annotations, left_on='question_id', right_on='question_id', how='right')
            df["image_id"] = df["image_id_x"]
            if not all(df["image_id_y"] == df["image_id_x"]):
                        print("There is something wrong with image_id")
            del df["image_id_x"]
            del df["image_id_y"]
            self.df = df
            self.n_samples = self.df.shape[0]

        elif dataset =="GQA":
                self.df = {}
                self.df["image_path"]=[]
                self.df["question"]=[]
                self.df["answers"]=[]
                self.vocab={}
                i=0
                with open('/raid/biplab/hassan/VQA_CLIP/vqa_common_gqa.txt', 'r') as file:
                        for line in file:
                            self.vocab[line[:-1]]=i
                            i+=1
                # print(self.vocab.keys())
                for answer in tqdm(data.values()):
                    question = answer["question"]
                    answers = answer["answer"]
                    # print(answers)
                    if answers in self.vocab.keys():
                        # print("here")
                        path = os.path.join(self.IMAGE_PATH[split][0],answer["imageId"]+".jpg")
                        path = os.path.join(root,path)
                        self.df["image_path"].append(path)
                        self.df["answers"].append(answers)
                        self.df["question"].append(question)
        elif dataset == 'VG':
            counts=0
            i=0
            self.vocab={}
            self.df = {}
            self.df["image_path"]=[]
            self.df["question"]=[]
            self.df["answers"]=[]
            with open('/raid/biplab/hassan/VQA_CLIP/vqa_common_vg.txt', 'r') as file:
                        for line in file:
                            self.vocab[line[:-1]]=i
                            i+=1
            leng = len(data)
            print(leng)
            for answer in tqdm(data):
                if split[:5] == 'train' and counts<(int)(leng*0.7):
                    for q in answer['qas']:
                        question = q["question"]
                        answers = q["answer"]
                        answers = preprocessing(answers)
                        if answers in self.vocab.keys():
                            path = os.path.join(self.IMAGE_PATH[split][0],str(q["image_id"])+".jpg")
                            path = os.path.join(root,path)
                            self.df["image_path"].append(path)
                            self.df["question"].append(question)
                            self.df["answers"].append(answers)
                elif split[:3] == 'val' and counts>=(int)(leng*0.7):
                    for q in answer['qas']:
                        question = q["question"]
                        answers = q["answer"]
                        answers = preprocessing(answers)
                        # print(answers)
                        if answers in self.vocab.keys():
                            # print("here")
                            path = os.path.join(self.IMAGE_PATH[split][0],str(q["image_id"])+".jpg")
                            path = os.path.join(root,path)
                            self.df["image_path"].append(path)
                            self.df["question"].append(question)
                            self.df["answers"].append(answers)
                counts+=1
    def __getitem__(self, index):
        image_path = self.df["image_path"][index]
        question = self.df["question"][index]
        if self.dataset == 'VQAv2' or self.dataset == 'VQAab':
            selected_answers = self.selection(self.df["answers"][index])
        else: 
            selected_answers = self.df["answers"][index]
        image_path = os.path.expanduser(os.path.join(self.root, image_path))
        img = Image.open(image_path).convert('RGB')
        img = preprocess(img)
        answer = torch.tensor(self.vocab[selected_answers])
        return {"img": img, "question": question, "answer": answer}
    def __len__(self):
        return len(self.df["answers"])