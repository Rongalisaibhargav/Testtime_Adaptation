from PIL import Image
import json
from tqdm import tqdm
import os
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

# path = '/raid/biplab/hassan/datasets/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json'
path = '/raid/biplab/hassan/datasets/vqa_abs/OpenEnded_abstract_v002_train2015_questions.json'
with open(path, 'r') as f:
            data = json.load(f)
root = '/raid/biplab/hassan/datasets/vqa_abs'
lists =[]
count=0
quid_skip=[]
captions=""
id_same =-1
for ques in tqdm(data["questions"]):
    image_id = ques["image_id"]
    # if image_id!=id_same:
    # print(image_id)
    # /raid/biplab/hassan/datasets/vqa_v2/train2014/COCO_train2014_000000000009.jpg
    # image_id = f"train2014/COCO_train2014_{image_id:012d}.jpg"
    image_id = f"train2015/abstract_v002_train2015_{image_id:012d}.png"
    image_path = os.path.join(root,image_id)    
    # text = "an image of"
    img = Image.open(image_path).convert('RGB')
    inputs = processor(img, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    caps = processor.decode(out[0], skip_special_tokens=True)
    # else: 
    #     caps = captions
    # id_same = image_id
    refined_ques={}
    refined_ques["caption"] = caps
    refined_ques["question"]= ques["question"]
    refined_ques["question_id"]= ques["question_id"]
    refined_ques["image_id"] = ques["image_id"]
    lists.append(refined_ques)
    # break
question_final ={}
question_final["questions"]=lists
# file_path = 'v2_OpenEnded_mscoco_val2014_captions.json'
file_path = "OpenEnded_abstract_v002_train2015_captions.json"
# print("questions_skipped ",quid_skip)
# Dump the dictionary to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(question_final, json_file, indent=4)

