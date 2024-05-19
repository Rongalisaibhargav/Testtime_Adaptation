from transformers import AutoTokenizer
import transformers
import torch
import json
import re
from tqdm import tqdm
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="cuda:1",
)
path = '/raid/biplab/hassan/datasets/vqa_abs/OpenEnded_abstract_v002_val2015_questions.json'
with open(path, 'r') as f:
            data = json.load(f)
lists =[]
count=0
quid_skip=[]
for ques in tqdm(data["questions"]):
    # if count>30:
    #       break
    refined_ques={}
    question = ques["question"]
    # Given global answer space is [jackal, red, elephant, Yes, No, chair, parrot, green]. if the question is "what is the color of jacket he is wearing ?" then all possible answer phrases to this question are ["The color of the jacket is green","The color of the jacket is jackal", "The color of the jacket is red", "The color of the jacket is elephant", "The color of the jacket is Yes", "The color of the jacket is No", "The color of the jacket is chair", "The color of the jacket is parrot", "The color of the jacket is green"]. Don't care about context of answer phrase and similarly give respective answer phrase for the following question. You should take only key words from the global answer space and frame the phrase according to the question
    # Given the global answer space is [jackal, red, elephant, Yes, No, chair, parrot, green, blue]. And the question is 'Is she wearing a long sleeve?' You need to give complete answer phrases like ['Yes, she is wearing a long sleeve', 'No, she is not wearing a long sleeve']. Do not combine two answers within a single phrase and each phrase should contain an answer from the global answer space and don't use words outside this context. What are all the possible meaningful answer phrases with distinct answer words for the question
    # Question1: What is the worker wearing? Phrase1: The worker is wearing <answer>, Question2: What is the biggest thing on the road? Phrase2: The biggest thing on the road is <answer>, Question3: How many animals are there in the picture? Phrase3: There are <answer> animals Question4: Where is the mobilephone? Pharse4: mobilephone is at <answer> Question5: Is the sky clear? Phrase5:  Yes, the sky is clear. Do not fill the <answer>, just leave it as a place holder. Please start the phrase with "Phrase:"
    input = f"""
            <<SYS>>
            Give phrase corresponding to the given question. Example: Question1: What is the worker wearing? Phrase1: The worker is wearing <answer>. Question2: What is the biggest thing on the road? Phrase2: The biggest thing on the road is <answer>. Question3: How many animals are there in the picture? Phrase3: There are <answer> animals. Question4: Where is the mobile phone? Phrase4: The mobile phone is at <answer>. Question5: Is the sky clear? Phrase5: Yes, the sky is clear. Question6: What color is the car? Phrase6: The car is <answer>. Question7: Who is sitting on the bench? Phrase7: <answer> is sitting on the bench. Question9: What is the woman holding? Phrase9: The woman is holding <answer>. Question10: How old is the building? Phrase10: The building is <answer> years old. Question11: Does the room have a window? Phrase11: Yes, the room has a window. Do not fill the <answer>, just leave it as a place holder with <answer> tag. Please start the phrase with "Phrase:"
            [INST]
            Question :{question}
            [/INST]\n

            Assistant:
        """
    sequences = pipeline(
        input,
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=400,
    )
    for seq in sequences:
        strs = seq['generated_text']
    # break
    # match = re.search(r'Phrase:\s*(.*)', strs, re.DOTALL)
    match = re.findall(r'Assistant:\s*Phrase:\s*(.*)', strs)
    if match:
        refined_ques["phrase"] = match[0]
        refined_ques["question"]= ques["question"]
        refined_ques["question_id"]= ques["question_id"]
        refined_ques["image_id"] = ques["image_id"]
        lists.append(refined_ques)
        count+=1
    else:
        quid_skip.append(ques["question_id"])
        continue

    # print(question)
    # print(match)
    # if match:
    #     captured_string = match.group(1)
    #     print(captured_string)
    # else:
    #     print("No match")
# print(strs)
question_final ={}
question_final["questions"]=lists
file_path = 'OpenEnded_abstract_v002_val2015_phrases.json'
print("questions_skipped ",quid_skip)
# Dump the dictionary to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(question_final, json_file, indent=4)
