from langchain_community.document_loaders import PyMuPDFLoader
import pymupdf4llm
import pathlib

loader = PyMuPDFLoader("attention.pdf")
data = loader.load()
#print(data)                                                                    #this is a list of dictionaries, each dictionary represents a page of the document
# convert the document to markdown
md_text = pymupdf4llm.to_markdown("attention.pdf")                              #This is a string of markdown text

# save the markdown text to a file
pathlib.Path("output.md").write_bytes(md_text.encode())
