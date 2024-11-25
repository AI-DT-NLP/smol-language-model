import json
from transformers import AutoTokenizer

def preprocess(input_path, output_path, tokenizer_name="gpt2", max_length=512):
    """
    Tokenize and preprocess dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    with open(input_path, "r") as f:
        data = json.load(f)

    preprocessed_data = []
    for item in data:
        text = item.get("text", "")
        tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        if len(tokens["input_ids"][0]) > 0:
            preprocessed_data.append({"input_ids": tokens["input_ids"][0].tolist()})

    with open(output_path, "w") as f:
        json.dump(preprocessed_data, f, indent=4)

    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess(
        input_path="./datasets/raw_dataset.json",
        output_path="./datasets/preprocessed/baseline.json"
    )
