from transformers import DetrImageProcessor, DetrForObjectDetection
from collections import defaultdict
from transformers import pipeline
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import openai
from openai import OpenAI
import requests
import torch
import os

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with the API key
client = OpenAI(api_key=openai_api_key)

# Define image URLs
image_urls = [
    "http://images.cocodataset.org/train2014/COCO_train2014_000000491269.jpg",
    "http://images.cocodataset.org/train2014/COCO_train2014_000000491274.jpg",
    "https://storage.googleapis.com/reka-annotate.appspot.com/vibe-eval/difficulty-normal_misc17_8e3edb121303f324.jpg"
]

# Initialize DETR processor and model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Lists to store detected objects, sentences, and final story lines
all_detected_objects = []
all_sentences = []

# Process each image
for image_url in image_urls:
    # Load the input image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Process the input image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Convert outputs to COCO API format
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]
    print("")
    # Extract detected objects and their frequencies
    detected_objects = defaultdict(int)
    for label in results["labels"]:
        detected_objects[model.config.id2label[label.item()]] += 1

    all_detected_objects.append(detected_objects)

  # Construct a prompt for GPT-3.5
    prompt = "You are a visual storyteller. You see a scene filled with: "
    for object_name, count in detected_objects.items():
    #    prompt += f"{count} {object_name}, "
         prompt += f"{object_name}, "
    prompt = prompt.rstrip(", ") + ". Generate a visual story sentence with these objects in 2 lines."

    # Generate a visual story using GPT-3.5
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": prompt},
        ]
    )

    # Store generated sentences
    sentence = completion.choices[0].message.content.split("\n")
    all_sentences.append(sentence)

# Print detected objects in each image
for i, detected_objects in enumerate(all_detected_objects):
    print(f"Objects detected in image {i+1}: {', '.join(detected_objects.keys())}\n")

# Print sentences generated for each image
for i, sentence in enumerate(all_sentences):
    print(f"\nSentences generated for image {i+1}:")
    for line in sentence:
        print(line)

# Combine sentences for story generation
story_prompt = "Please craft a compelling and cohesive visual story that integrates all three provided sentences. The narrative should be interconnected and dependent, with each sentence serving a pivotal role in the story's progression. The story should be presented in a single, well-structured paragraph of at least 30 lines, with no newlines or breaks."
for i, sentence in enumerate(all_sentences):
    story_prompt += f"\nScene {i+1}: {sentence[0]}"

story_prompt += "\nPlease craft a compelling and cohesive visual story that integrates all three provided sentences. The narrative should be interconnected and dependent, with each sentence serving a pivotal role in the story's progression. The story should be presented in a single, well-structured paragraph of at least 30 lines, with no newlines or breaks."
# Generate a visual story using GPT-3.5
story_completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "system", "content": story_prompt},
    ]
)

# Store generated story
story = story_completion.choices[0].message.content.split("\n")

# Print the generated visual story
print("\nGenerated Story:")
for line in story[:10]:
    print(line)