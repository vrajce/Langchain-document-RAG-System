from langchain_community.document_loaders import DirectoryLoader , TextLoader , UnstructuredMarkdownLoader
from langchain_community.document_loaders.base_o365 import CHUNK_SIZE
from langchain_text_splitters import MarkdownHeaderTextSplitter , RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings , HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


loader = DirectoryLoader(
    './docs' , 
    glob="**/*.mdx" ,
    loader_cls=UnstructuredMarkdownLoader, 
    show_progress=True , 
    loader_kwargs={'encoding':'utf-8'}
)

# print(loader)
headers_to_split_on=[("#", "Header 1"),("##", "Header 2"),("###", "Header 3")]
markDownSplitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

md_chunks=[]

for doc in loader.lazy_load():
    
    splits = markDownSplitter.split_text(doc.page_content)
    
    for split in splits:
        split.metadata.update(doc.metadata)
        source = split.metadata['source'].lower()
        print(f"\n\n\n\n\n\n\n\n\n\n\n\n\n {source}")
        
        # Check specific libraries FIRST
        if 'langsmith' in source:
            ecosystem = 'langsmith'
        elif 'langgraph' in source:
            ecosystem = 'langgraph'
        elif 'integrations' in source:
            ecosystem = 'integrations'
        elif 'concepts' in source:
            ecosystem = 'concepts'
        elif 'javascript' in source:
            ecosystem = 'javascript'
        elif 'python' in source:
            ecosystem = 'langchain'  # The 'python' folder is usually the main LangChain SDK
        elif 'contributing' in source:
            ecosystem = 'contributing'
        else:
            ecosystem = 'general'
        
        split.metadata['ecosystem'] = ecosystem
        md_chunks.append(split)
        print(f"\n\n\n\n\n\n\n : split :  {split}")

    

secondarySplitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
)

final_splitted = secondarySplitter.split_documents(md_chunks)
print(f"✅ Final Production Chunks: {len(final_splitted)}")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(final_splitted, embedding)
Db_name = 'faiss_index_react'
vectorstore.save_local(
    Db_name 
)