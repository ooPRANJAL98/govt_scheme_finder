# ingest.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def ingest_files():
    print("Loading text files...")
    loader = DirectoryLoader(
        "schemes_text/",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    if not documents:
        print("No .txt files found in schemes_text/")
        return

    print(f"Loaded {len(documents)} files. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    print(f"Done! {len(chunks)} chunks stored in ChromaDB.")
    for doc in documents:
        print(f"  - {doc.metadata.get('source', 'unknown')}")

if __name__ == "__main__":
    ingest_files()