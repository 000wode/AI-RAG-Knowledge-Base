from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

loader = TextLoader("knowledge_base/ai_intro.txt", encoding="utf-8")
documents = loader.load()
print("✅ 已加载文档")

# 手动切分文本，不依赖 langchain.text_splitter
text = documents[0].page_content
chunks = []
chunk_size = 200
for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i+chunk_size])
print(f"✅ 已切分为 {len(chunks)} 个文本块")

from langchain.schema import Document
doc_chunks = [Document(page_content=c) for c in chunks]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(documents=doc_chunks, embedding=embeddings, persist_directory="./chroma_db")
vector_store.persist()
print("✅ 向量库已构建并保存到 ./chroma_db/")