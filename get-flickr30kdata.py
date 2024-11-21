from datasets import load_dataset
import os
import json

dataset = load_dataset("nlphuji/flickr30k")

dataset_path = "./data/flickr-dataset"
images_path = os.path.join(dataset_path, "images")
captions_file = os.path.join(dataset_path, "captions.json")

os.makedirs(images_path, exist_ok=True)

captions_data = []
for idx, example in enumerate(dataset["test"]):
    image_filename = example["filename"]
    image_filepath = os.path.join(images_path, image_filename)
    example["image"].save(image_filepath)

    for caption in example["caption"]:
        conversations = []
        conversations.append({
            "from": "human",
            "value": "<image>\nDescribe the image."
        })
        conversations.append({
            "from": "gpt",
            "value": caption
        })

        captions_data.append({
            "id": example["img_id"],
            "image": [image_filename],
            "conversations": conversations
        })
    print('processed', idx, example)


with open(captions_file, "w") as f:
    json.dump(captions_data, f, indent=4)

print(f"Dataset successfully downloaded and saved in: {dataset_path}")