from langchain.llms import HuggingFacePipeline
from hf_pipeline import generate_text

llm = HuggingFacePipeline(pipeline=generate_text)

if __name__ == '__main__':
    res = llm(prompt="Explain me the difference between Data Lakehouse and Data Warehouse")
    print(res)
