import requests
from bs4 import BeautifulSoup
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# OpenAIを使わずにHuggingFaceの埋め込みを使用
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

import os
import streamlit as st

# Webページを取得する関数
def fetch_web_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        text = ' '.join([p.get_text() for p in soup.find_all("p")])
        return text
    else:
        return None

# 記事のテキストを保存し、RAGに登録
def index_web_contents(urls, save_dir="web_articles"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for idx, url in enumerate(urls):
        content = fetch_web_content(url)
        if content:
            filename = os.path.join(save_dir, f"article_{idx}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
    
    # LlamaIndexに登録
    documents = SimpleDirectoryReader(save_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()
    
    return index

# ChatGPTとの連携
def ask_chatbot_with_web(question, index):
    from openai import OpenAI
    from llama_index.core.query_engine import RetrieverQueryEngine
    query_engine = RetrieverQueryEngine(index)
    context = query_engine.query(question)
    
    response = OpenAI.ChatCompletion.create(
        model="gpt-4o mini",
        messages=[
            {"role": "system", "content": f"Use this context: {context}"},
            {"role": "user", "content": question}
        ]
    )
    
    return response["choices"][0]["message"]["content"]

# Streamlit UI
st.title("WebリンクをRAGに登録し、AIに質問")

urls = st.text_area("Webリンクを入力（改行で複数追加）")
question = st.text_input("AIに質問する内容")

if st.button("登録＆検索"):
    url_list = [url.strip() for url in urls.split("\n") if url.strip()]
    if url_list:
        index = index_web_contents(url_list)
        if question:
            answer = ask_chatbot_with_web(question, index)
            st.write("### 回答:")
            st.write(answer)
        else:
            st.write("質問を入力してください。")
    else:
        st.write("有効なURLを入力してください。")
