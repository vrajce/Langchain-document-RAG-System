from langchain_huggingface import HuggingFaceEmbeddings , HuggingFaceEndpointEmbeddings


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

query = "hello , i'm a student"

print(embedding.embed_query(query))