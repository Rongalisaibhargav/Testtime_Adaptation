from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load the pretrained T5 model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Define the list of possible global answer spaces
answer_spaces = ["jackal", "red", "elephant", "Yes", "No", "chair", "parrot", "green", "blue"]

# Define a function to generate answer phrases
def generate_answer_phrases(question, answer_spaces):
    input_ids = tokenizer.encode(question, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    generated_answer = tokenizer.decode(outputs, skip_special_tokens=True)

    return generated_answer

# Example usage
questions = [
    "What is the worker wearing? The worker is wearing [mask]. Where is he looking? <extra_id_0>."
]

for question in questions:
    answer_phrases = generate_answer_phrases(question, answer_spaces)
    print(f"Question: {question}")
    for phrase in answer_phrases:
        print(f"Answer: {phrase}")
    print()
