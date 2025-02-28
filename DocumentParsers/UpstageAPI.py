#DOCUMENT PARSER
# pip install -qU langchain-core langchain-upstage
import os
from langchain_upstage import UpstageDocumentParseLoader
 
UPSTAGE_API_KEY =os.getenv("UPSTAGE_API_KEY")
file_path = "invoice.png"
loader = UpstageDocumentParseLoader(file_path, ocr="force")   #forces ocr to be applied, even if file already contains extractable text
pages = loader.load()                                         #returns a list of strings, each string representing a page of the document
for page in pages:
    print(page)

#----------------------------------------------------------------------------------------------------------------------------------------------------
#DOCUMENT OCR 
# pip install requests
 
import requests, os
 
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
filename = "hello.webp"
 
url = "https://api.upstage.ai/v1/document-ai/ocr"
headers = {"Authorization": f"Bearer {UPSTAGE_API_KEY}"}
 
files = {"document": open(filename, "rb")}
response = requests.post(url, headers=headers, files=files)
 
print(response.json())

#------------------------------------------------------------------------------------------------------------------------------------------------------

