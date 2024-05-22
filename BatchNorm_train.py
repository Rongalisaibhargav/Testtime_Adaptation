from data.dataset import VQAv2Dataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import os
import re
import clip
from tqdm import tqdm

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", device=device)

# Ensure tokenizers parallelism is disabled for stable processing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Mean and standard deviation for normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_workers = 4
image_size = 224
batch_size = 1

# Image transformations including normalization
train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)  # Normalization is included
])

# VQAv2 dataset for validation
val_dataset = VQAv2Dataset('/raid/biplab/hassan/datasets/vqa_v2', 'val', 'VQAv2', transform=train_transform)
val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

# Function to generate yes/no phrases
def generate_yes_no_phrases(phrase):
    pattern = r"\b(is|was|has been|are|were)\b"
    match = re.search(pattern, phrase, re.IGNORECASE)
    
    if match:
        verb = match.group(0)
        start, end = match.span(0)
        yes_phrase = "Yes, " + phrase
        no_phrase = phrase[:start] + "is not" + phrase[end:]
        
        return [yes_phrase, "No, " + no_phrase]
    else:
        return ["Yes, " + phrase, "No, " + phrase]

# Condition check function
condition = lambda x: x == 'yes' or x == 'no'

# Initialize vocabulary and reverse mapping
i = 0
vocab = {}
reverse = {}
vocab_path = '/raid/biplab/hassan/VQA_CLIP/vqa_common_ab.txt'
with open(vocab_path, 'r') as file:
    for line in file:
        line = line.strip()
        vocab[line] = i
        reverse[i] = line
        i += 1

# Initialize counters for accuracy calculation
correct = 0
total = 0

# Iterate through validation dataset
for data in tqdm(val_loader):
    img = data["img"]
    ques = data["phrase"]
    ans = data["answer"]
    img = img.to(device)
    
    # Encode image using CLIP
    with torch.no_grad():
        image_features = model.encode_image(img)
    
    # Generate phrases based on the question and answer type
    if ans[0] == 'yes' or ans[0] == 'no':
        phrases = generate_yes_no_phrases(ques[0])
    else:
        pattern = r"(.*?)<answer>(.*?)| "
        if re.match(pattern, ques[0]):
            phrases = [ques[0].replace("<answer>", c) for c in vocab.keys()]
    
    try:
        # Tokenize and encode text using CLIP
        inputs = clip.tokenize(phrases).to(device)
        with torch.no_grad():
            text_features = model.encode_text(inputs)
        
        # Compute similarity between image and text features
        result = torch.mm(image_features, text_features.t())
        out = torch.argmax(result, dim=1).cpu().item()
        
        # Convert numerical output to text for comparison
        conv = {0: "yes", 1: "no"}
        total += 1
        
        # Check if prediction matches the answer
        if ans[0] == 'yes' or ans[0] == 'no':
            if conv[out] == ans[0]:
                correct += 1
        else:
            if out == vocab[ans[0]]:
                correct += 1
        
        # Print accuracy every 100 steps
        if total % 100 == 0:
            print("Accuracy:", correct / total)
    except Exception as e:
        continue
