from datasets import load_dataset
import json

def download_dataset(dataset_name, save_path):
    """
    Download and save dataset in JSON format.
    
    Args:
        dataset_name (str): Name of the dataset to download.
        save_path (str): Path to save the dataset.
    """
    dataset = load_dataset(dataset_name) 
    dataset = dataset["train"]  # Use the training split

    # Save to JSON file
    data = [{"text": item["text"]} for item in dataset]
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    # TODO: find pre-train dataset.
    download_dataset(dataset_name="codeparrot/code-textbook", save_path="./datasets/raw_dataset.json")
