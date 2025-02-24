import requests
from bs4 import BeautifulSoup
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from openai import OpenAI
import os
import streamlit as st
import json

# OpenAIの埋め込みを使用
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")


# 保存用のJSONファイル
data_file = "saved_links.json"

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

# 保存済みのリンクを読み込む
def load_saved_links():
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# リンクを保存する
def save_links(category, links):
    data = load_saved_links()
    if category not in data:
        data[category] = []
    data[category].extend(links)
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# Streamlit UI
st.title("WebリンクをRAGに登録し、AIに質問")

categories = st.text_input("カテゴリーを入力")
urls = st.text_area("Webリンクを入力（改行で複数追加）")
question = st.text_input("AIに質問する内容")

if st.button("登録＆検索"):
    url_list = [url.strip() for url in urls.split("\n") if url.strip()]
    if url_list and categories:
        save_links(categories, url_list)
        index = index_web_contents(url_list)
        if question:
            answer = ask_chatbot_with_web(question, index)
            st.write("### 回答:")
            st.write(answer)
        else:
            st.write("質問を入力してください。")
    else:
        st.write("有効なURLとカテゴリーを入力してください。")

# ギャラリー表示
data = load_saved_links()
st.write("### 保存済みのリンク")
for category, links in data.items():
    with st.expander(category):
        for link in links:
            st.markdown(f"[🔗 {link}]({link})")
