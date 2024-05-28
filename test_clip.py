from data.dataset import VQAv2Dataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from transformers import AutoModelWithLMHead, AutoTokenizer
import os
import re
from PIL import Image
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_workers = 4
image_size = 224
batch_size= 1

train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor(mean),
        std=torch.tensor(std))
])

img_path = '/raid/biplab/hassan/datasets/vqa_v2/train2014/COCO_train2014_000000112059.jpg'
phrases = ['The guitar is on the right side of the monitor','The guitar is on the left side of the monitor']
# phrases = ['There are 4 black cats and 6 brown donkey.', 'There are 4 black cats and 3  brown donkey.','There are 4 black horses and 4  brown horses.','There are 4 black cats and 5  brown donkey.']
# phrases_class = ['giraffe', 'toothbrush', 'dog','elephant']
phrases_class =['Yes, there are toothbrushes.', 'No, there are no toothbrushes.']

inputs = clip.tokenize(phrases).to(device)
img = Image.open(img_path).convert('RGB')
img = preprocess(img).to('cuda')
# print(img.shape)
with torch.no_grad():
    image_features = model.encode_image(img.unsqueeze(0))
    text_features = model.encode_text(inputs)
# print(image_features.shape)
# print(text_features.shape)
result = torch.mm(image_features,text_features.t())
print(result)
print(torch.argmax(result))
inputs = clip.tokenize(phrases_class).to(device)
with torch.no_grad():
    text_features = model.encode_text(inputs)
# print(image_features.shape)
# print(text_features.shape)
result = torch.mm(image_features,text_features.t())
print(result)
print(torch.argmax(result))