import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torch
import torch.nn.functional as F
# from lavis.models import load_model_and_preprocess
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
import numpy as np
import re
# import cv2
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from tqdm import tqdm
import clip
def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
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
class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [self.preprocess(self.base_transform(x)) for _ in range(self.n_views)]
        return [image] + views
@DATASET_REGISTRY.register()
class VQAv2(DatasetBase):
    IMAGE_PATH = {
        "train": ("train2014","v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json","train2014"),
        "val": ("val2014","v2_OpenEnded_mscoco_val2014_phrases.json", "v2_mscoco_val2014_annotations.json","val2014"),
        "train_ab": ("train2015","OpenEnded_abstract_v002_train2015_questions.json", "abstract_v002_train2015_annotations.json","train2015"),
        "val_ab": ("val2015","OpenEnded_abstract_v002_val2015_phrases.json", "abstract_v002_val2015_annotations.json","val2015"),
        "testdev": ("test2015", "v2_OpenEnded_mscoco_test-dev2015_questions.json"),
        "test": ("test2015", "v2_OpenEnded_mscoco_test2015_questions.json"),
        "train_gqa":("images","train_balanced_questions.json"),
        "val_gqa":("images","val_balanced_questions.json"),
        "train_vg":("images","question_answers.json"),
        "val_vg":("images","question_answers.json")}
    def __init__(self,cfg):
        # split for self.dataset
        # train,val for VQAv2 self.dataset
        # train_ab,val_ab for VQA Abs self.dataset
        # train_gqa,val_gqa for GQA self.dataset
        # train_vg, val_vg for VG self.dataset
        self.split = cfg.DATASET.SPLIT
        # path to the root folder of the self.dataset
        self.root = cfg.DATASET.ROOT
        self.vocab={}
        # self.dataset name : VQAv2, VQA Ab, GQA and VG
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        # data_transform = transforms.Compose([
        #         transforms.Resize(224, interpolation=BICUBIC),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ])
        base_transform = transforms.Compose([
                transforms.Resize((224,224), interpolation=BICUBIC),
                transforms.GaussianBlur(3),
                transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1))])
        preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=63, 
                                            augmix=False)
        # batchsize = 1
        self.dataset = cfg.DATASET.DATA
        self.transform =  data_transform
        self.selection = most_common_from_dict
        path = os.path.expanduser(os.path.join(self.root, self.IMAGE_PATH[self.split][1]))

        with open(path, 'r') as f:
            data = json.load(f)
        if self.dataset=="VQAv2" or self.dataset=="VQAab":
            df = pd.DataFrame(data["questions"])
            if self.dataset=="VQAv2":
                df["image_path"] = df["image_id"].apply(
                        lambda x: f"{self.IMAGE_PATH[self.split][0]}/COCO_{self.IMAGE_PATH[self.split][3]}_{x:012d}.jpg")
            elif self.dataset=="VQAab":
                    df["image_path"] = df["image_id"].apply(
                        lambda x: f"{self.IMAGE_PATH[self.split][0]}/abstract_v002_{self.IMAGE_PATH[self.split][3]}_{x:012d}.png")
            path = os.path.expanduser(os.path.join(self.root, self.IMAGE_PATH[self.split][2]))
            # print(df.keys())
            with open(path, 'r') as f:
                        data = json.load(f)
            df_annotations = pd.DataFrame(data["annotations"])
            # pattern = r"(.*?)<answer>(.*?)| "
            # df = df[df['phrase'].str.contains(pattern, regex=True, na=False)]
            # print(df['phrase'][:10])
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
            df =df[df['image_id_y'] == df['image_id_x']].reset_index()
            if not all(df["image_id_y"] == df["image_id_x"]):
                        print("There is something wrong with image_id")
            del df["image_id_x"]
            del df["image_id_y"]
            self.df = df
            self.n_samples = self.df.shape[0]

        elif self.dataset =="GQA":
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
                        path = os.path.join(self.IMAGE_PATH[self.split][0],answer["imageId"]+".jpg")
                        path = os.path.join(self.root,path)
                        self.df["image_path"].append(path)
                        self.df["answers"].append(answers)
                        self.df["question"].append(question)
        elif self.dataset == 'VG':
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
                if self.split[:5] == 'train' and counts<(int)(leng*0.7):
                    for q in answer['qas']:
                        question = q["question"]
                        answers = q["answer"]
                        answers = preprocessing(answers)
                        if answers in self.vocab.keys():
                            path = os.path.join(self.IMAGE_PATH[self.split][0],str(q["image_id"])+".jpg")
                            path = os.path.join(self.root,path)
                            self.df["image_path"].append(path)
                            self.df["question"].append(question)
                            self.df["answers"].append(answers)
                elif self.split[:3] == 'val' and counts>=(int)(leng*0.7):
                    for q in answer['qas']:
                        question = q["question"]
                        answers = q["answer"]
                        answers = preprocessing(answers)
                        # print(answers)
                        if answers in self.vocab.keys():
                            # print("here")
                            path = os.path.join(self.IMAGE_PATH[self.split][0],str(q["image_id"])+".jpg")
                            path = os.path.join(self.root,path)
                            self.df["image_path"].append(path)
                            self.df["question"].append(question)
                            self.df["answers"].append(answers)
                counts+=1
        tr =[]
        for i in range(10):
            # tr = Datum(im
            if self.dataset == 'VQAv2' or self.dataset == 'VQAab':
                image_path = os.path.expanduser(os.path.join(self.root, self.df["image_path"][i]))
                selected_answers = self.selection(self.df["answers"][i])
            print(image_path)
            tr.append(Datum(impath=image_path , label=self.vocab[selected_answers], classname=selected_answers))
        super().__init__(train_x=tr, val=tr, test=tr)
    def __getitem__(self, index):
        image_path = self.df["image_path"][index]
        question = self.df["question"][index]
        phrase = self.df["phrase"][index]
        ans = self.df["answers"][index]
        # test_phrases= []
        if self.dataset == 'VQAv2' or self.dataset == 'VQAab':
            selected_answers = self.selection(ans)
        else: 
            selected_answers = ans     
        image_path = os.path.expanduser(os.path.join(self.root, image_path))
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        answer = selected_answers
        # print(img[0].shape)
        return {"img": img, "question": question, "answer": torch.tensor(self.vocab[answer]), "phrase": phrase}
    def __len__(self):
        return len(self.df["answers"])