import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")

# 1. Setup - Replace with your key
# (Or set it in your .env file as HUGGINGFACEHUB_API_TOKEN)


# 2. Load the "Brain" (FAISS Index)
print("🔌 Loading the Frozen Brain...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_index_react", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 3. Setup the LLM (Mistral-7B via Hugging Face API)
print("🧠 Connecting to the LLM...")
repo_id = "mistralai/Mistral-7B-Instruct-v0.3" 

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

# 4. Create the "Answer Template"
# This tells the LLM how to behave.
prompt_template = """
You are a LangChain Expert Support Bot.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

# 5. Build the Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "Stuff" means: Put all retrieved context into the prompt
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# 6. The Chat Loop
print("\n🤖 Bot is Ready! (Type 'exit' to stop)")
while True:
    query = input("\nUser: ")
    if query.lower() == "exit":
        break
    
    # Run the query
    print("Thinking...")
    response = qa_chain.invoke({"query": query})
    
    # Print the Answer
    print(f"\n🤖 AI: {response['result']}")
    
    # Print Sources (Like the Pro Bot!)
    print("\n📚 Sources:")
    for doc in response['source_documents']:
        print(f"- {doc.metadata['source']} ({doc.metadata['ecosystem']})")