import requests, os
import numpy as np  # For cosine similarity
from langchain_upstage import UpstageEmbeddings

# API key setup
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY") or "up_*************************9tDx"
filename = "cars.pdf"

# OCR request
url = "https://api.upstage.ai/v1/document-ai/ocr"
headers = {"Authorization": f"Bearer {UPSTAGE_API_KEY}"}
files = {"document": open(filename, "rb")}
response = requests.post(url, headers=headers, files=files)
parsed_data = response.json()

# Debugging output (optional)
# print("OCR Response:", parsed_data)

# Extract text based on response structure
if "pages" in parsed_data:
    first_page = parsed_data["pages"][0]
    text_key = "text"  # Adjust this based on actual key (e.g., "text" from previous debug)
    if text_key in first_page:
        text_list = [page[text_key] for page in parsed_data["pages"]]
    else:
        raise KeyError(f"No '{text_key}' key found. Available keys: {first_page.keys()}")
elif "text" in parsed_data:
    text_list = [parsed_data["text"]]
else:
    raise ValueError("No 'pages' or 'text' key found in OCR response")

print("Extracted Text:", text_list)

# Initialize embeddings
embeddings = UpstageEmbeddings(
    api_key=UPSTAGE_API_KEY,
    model="embedding-query"
)

# Generate embeddings for document text and query
doc_result = embeddings.embed_documents(text_list)
query_result = embeddings.embed_query("read the document content, and tell the user what is present")

# Compute cosine similarity between query and document embeddings
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

similarities = [cosine_similarity(query_result, doc_embedding) for doc_embedding in doc_result]

# Find the most similar document segment
most_similar_idx = np.argmax(similarities)
most_similar_text = text_list[most_similar_idx]
similarity_score = similarities[most_similar_idx]

# Generate a response
if similarity_score > 0.5:  # Arbitrary threshold for relevance
    response = f"The document contains: '{most_similar_text}'"
else:
    response = "The document content doesn't clearly match the query."

print("Response to 'What does the document contain?':", response)
