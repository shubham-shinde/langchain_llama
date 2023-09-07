from langchain.chains import ConversationalRetrievalChain
from langchain_integration import llm
from vector_store import vectorstore

chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(),
    return_source_documents=True
)

if __name__ == '__main__':
    chat_history = []
    query = "What is Data lakehouse architecture in Databricks?"
    result = chain({'question': query, 'chat_history': chat_history})
    print(result['answer'])