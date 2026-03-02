from langchain_community.document_loaders import DirectoryLoader , TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

loader = DirectoryLoader('./docs' , glob="**/*.mdx" ,loader_cls=TextLoader, show_progress=True , loader_kwargs={'encoding':'utf-8'})
raw_docs = loader.load()

print(f'length of raw docs: {len(raw_docs)}')

# print(raw_docs[5])
headers_to_split_on=[("#", "Header 1"),("##", "Header 2"),("###", "Header 3")]
markDownSplitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

md_chunks=[]

for doc in raw_docs:
    splits = markDownSplitter.split_text(doc)
    print(f)
    for split in splits:
        md_chunks.append(split)
    
    