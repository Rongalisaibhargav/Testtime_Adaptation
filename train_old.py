from data.dataset import VQAv2Dataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from transformers import AutoModelWithLMHead, AutoTokenizer
import os
import re
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
# model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
# tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")
# model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing").to('cuda:0')

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

# # strs = "What is the worker wearing? The worker is wearing [mask]. " 
# strs = "What is the worker wearing? The worker is wearing [mask]."
# input_ids = tokenizer("What is the worker wearing? The worker is wearing [mask]. Where is he looking? <extra_id_0>", return_tensors="pt",padding=True,max_length=200).input_ids
# outputs = model.generate(input_ids)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# input_ids = tokenizer("paraphrase: What is the species of the plant?", return_tensors="pt",max_length=50).input_ids

# sequence_ids = model.generate(input_ids,num_return_sequences=5,num_beams=5)
# sequences = tokenizer.batch_decode(sequence_ids)
# print(sequences)

# input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids

val_dataset = VQAv2Dataset('/raid/biplab/hassan/datasets/vqa_v2','val','VQAv2',transform=train_transform)
val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
def generate_yes_no_phrases(phrase):
        # print(phrase)
        pattern = r"\b(is|was|has been|are|were)\b"
        match = re.search(pattern, phrase, re.IGNORECASE)
        
        if match:
            verb = match.group(0)
            start, end = match.span(0)
            yes_phrase = "Yes, " + phrase
            no_phrase = phrase[:start] + "is not" + phrase[end:]
            
            return [yes_phrase, "No, " + no_phrase]
        else:
            return "Yes, " + phrase, "No, " + phrase
condition = lambda x: x=='yes' | x=='no'
i=0
vocab={}
reverse ={}
vocab_path = '/raid/biplab/hassan/VQA_CLIP/vqa_common_ab.txt'
with open(vocab_path, 'r') as file:
    for line in file:
        line = line.replace('\n','')
        vocab[line]=i
        reverse[i]=line
        i+=1
correct=0
total=0
for data in tqdm(val_loader):
    # print(data.keys())
    # if count>10:
    #     break
    
    img = data["img"]
    ques = data["phrase"]
    ans = data["answer"]
    img = img.to('cuda')
    with torch.no_grad():
                image_features = model.encode_image(img)
    # ind = data["answer"] 
    
    if ans[0]=='yes' or ans[0]=='no':
        phrases = generate_yes_no_phrases(ques[0])
    else:
        pattern = r"(.*?)<answer>(.*?)| "
        # print(ques)
        if re.match(pattern,ques[0]):
            phrases=[]
            for c in vocab.keys():
                phrases.append(ques[0].replace("<answer>",c))
    try:
        inputs = clip.tokenize(phrases).to(device)
        with torch.no_grad():
            image_features = model.encode_image(img)
            text_features = model.encode_text(inputs)
        # print(image_features.shape)
        # print(text_features.shape)
        result = torch.mm(image_features,text_features.t())
        out =  torch.argmax(result,dim=1).cpu().item()
        conv ={}
        total+=1
        conv[0]="yes"
        conv[1]="no"
        if ans[0]=='yes' or ans[0]=='no':
            if conv[out]==ans[0]:
                correct+=1
            print("---")
            print(ques)
            print(ans[0])
            print(reverse[out])
            # print(out)
            print("---")
        else: 
            # print(ques)
            print("---")
            print(ques)
            print(ans[0])
            print(reverse[out])
            # print(out)
            print("---")
            
            if out == vocab[ans[0]]:
                correct+=1
        if total%100==0:
            print("Accuracy ",correct/total)
    except Exception as e:
        continue
    if total%9==0 and total!=0:
        break
    # print("-------------------------------------------")
    # count+=1




