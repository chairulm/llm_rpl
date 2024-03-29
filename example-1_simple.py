from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from getpass import getpass
import warnings
import torch
import gc


def run_myllm():

    #Template Prompt Text Wrapper 
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate.from_template(template)
    
    #Put the Model Here
    model_id = "01-ai/Yi-6B"
    #model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    #Using Transformer Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )

    #Pipeline Useage with Tokenizer
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        repetition_penalty=1.2,
        top_p=0.4,
        temperature=0.4,
        max_new_tokens=1000
    )
    
    #Langchain pipeline
    gpu_llm = HuggingFacePipeline(pipeline=pipe)
    
    #Langchain Chain
    gpu_chain = prompt | gpu_llm
    question = "Write a report the life of Thomas Jefferson and a separate report John Hopkins. Each report must be >
    print(gpu_chain.invoke({"question": question}),end="")
    print("\n----------------------\n")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    run_myllm()
