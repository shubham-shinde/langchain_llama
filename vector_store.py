from langchain.embeddings import HuggingFaceBgeEmbeddings
from document_loader import get_all_splits
from langchain.vectorstores import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}

embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

vectorstore = FAISS.from_documents(get_all_splits(), embeddings)