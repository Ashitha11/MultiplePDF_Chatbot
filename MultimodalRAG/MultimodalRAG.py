#Multimodal RAG --> retreiving text summaries from the images + creating vector embeddings 

import os
import requests
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment or .env file.")

# Create necessary directories
input_path_image = Path("input_images")
data_path = Path("mixed_wiki")

for path in [input_path_image, data_path]:
    if not path.exists():
        Path.mkdir(path)

print("Environment created successfully!")

# Display the images
image_paths = [str(os.path.join("./input_images", img_path)) for img_path in os.listdir("./input_images") if os.path.isfile(os.path.join("./input_images", img_path))]

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            images_shown += 1
    plt.show()

plot_images(image_paths)

# Generate text descriptions for the images
image_documents = SimpleDirectoryReader("./input_images").load_data()

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    max_new_tokens=1500
)

response = openai_mm_llm.complete(
    prompt="Generate detailed text description for each image.",
    image_documents=image_documents,
)

print("Generated Descriptions:")
print(response)

# Split the response into individual text chunks
description_texts = response.text.split("\n\n")
if len(description_texts) != len(image_paths):
    print(f"Warning: Number of descriptions ({len(description_texts)}) does not match number of images ({len(image_paths)}). Adjusting...")
    description_texts = description_texts[:len(image_paths)]  # Truncate or pad if necessary

# Create Document objects for each text chunk with metadata
documents = [
    Document(
        text=desc.strip(),
        metadata={"image_path": img_path, "source": "image_description"}
    )
    for desc, img_path in zip(description_texts, image_paths)
]

# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbedding(
    api_key=OPENAI_API_KEY,
    model="text-embedding-ada-002"
)

# Create a vector store index with the embeddings
vector_index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embedding_model
)

# Print the embeddings by querying the index or using the embedding model directly
print("\nVector Embeddings:")
for i, doc in enumerate(documents):
    # Generate the embedding for the document text directly using the embedding model
    embedding = embedding_model.get_text_embedding(doc.text)
    print(f"Embedding for Document {i+1} (Image: {doc.metadata['image_path']}):")
    print(f"Text: {doc.text[:100]}...")
    print(f"Shape: {np.array(embedding).shape}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Norm: {np.linalg.norm(embedding):.4f}\n")

# Save the vector store to disk for later use
vector_index.storage_context.persist(persist_dir="./vector_db")
print("Vector store saved to ./vector_db")

# Optional: Query the vector store to test retrieval
query_engine = vector_index.as_query_engine()
test_query = "What are the key specifications mentioned in the images?"
query_response = query_engine.query(test_query)
print("\nTest Query Response:")
print(query_response)
