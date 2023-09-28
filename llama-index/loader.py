from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, ServiceContext, set_global_service_context
from llama_index.llms import HuggingFaceLLM
from IPython.display import display, Markdown
from llama_index.prompts.prompts import SimpleInputPrompt

# prepare the template we will use when prompting the AI
system_prompt = """<<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
"""

query_wrapper_prompt = SimpleInputPrompt("[INST]{prompt}[/INST]")

def load():
    llm = HuggingFaceLLM(
        context_window=4096, 
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        model_name="NousResearch/Llama-2-7b-chat-hf",
        # model_name="./models/llama-2-7b-chat.Q4_K_M.gguf",
        # model="local:./models/llama-2-7b-chat.Q4_K_M.gguf",
        device_map="auto",
        stopping_ids=[50278, 50279, 50277, 1, 0],
        tokenizer_kwargs={"max_length": 4096},
        # uncomment this if using CUDA to reduce memory usage
        # model_kwargs={"torch_dtype": torch.float16}
    )
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024, chunk_overlap=128)

    document = SimpleDirectoryReader(input_files=["data/Business Conduct.pdf"]).load_data()
    index = VectorStoreIndex.from_documents(document, service_context=service_context) # this is just in memory
    index.storage_context.persist() # this enables storing vector store on disk

    storage_context = StorageContext.from_defaults(persist_dir="./llama-index/storage") # this is the default storage context
    index = load_index_from_storage(storage_context=storage_context) # this loads the index from disk
    query_engine = index.as_query_engine()
    # query_chat_bot = index.as_chat_bot() // this gives memory capabilties to the chat bot
    response = query_engine.search("what is Legal Holds?")
    display(Markdown(f"<b>{response}</b>"))

if __name__ == '__main__':
    load()