import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

# ------------------------------
# 1. 配置页面
# ------------------------------
st.set_page_config(page_title="智慧图书助手", layout="wide")
st.title("📚 智慧图书助手")

# ------------------------------
# 2. 加载数据与模型 (使用缓存提高性能)
# ------------------------------
@st.cache_resource
def load_data():
    # 优先使用包含聚类结果的文件，否则使用原始文件
    if os.path.exists("books_with_clusters.xlsx"):
        df = pd.read_excel("books_with_clusters.xlsx")
        st.info("已加载聚类结果文件 books_with_clusters.xlsx")
    else:
        df = pd.read_excel("books_dataset.xlsx")
        st.warning("未找到 books_with_clusters.xlsx，将只使用原始数据，聚类功能可能不完整")
        # 若没有聚类列，则手动添加一个占位列
        if "cluster" not in df.columns:
            df["cluster"] = -1
    # 确保简介列名称统一
    if "书本简介" in df.columns:
        df.rename(columns={"书本简介": "书本简介"}, inplace=True)
    elif "书本简介 " in df.columns:
        df.rename(columns={"书本简介 ": "书本简介"}, inplace=True)
    return df

@st.cache_resource
def load_embeddings():
    if os.path.exists("book_embeddings.npy"):
        return np.load("book_embeddings.npy")
    else:
        st.error("未找到 book_embeddings.npy 文件，请确认文件存在。")
        return None

@st.cache_resource
def load_model():
    # 尝试加载本地模型，失败则从 HuggingFace 下载
    local_model_path = r"C:\Users\huawei\Desktop\1.py\xiangsidu"  # 原路径
    if os.path.exists(local_model_path):
        try:
            model = SentenceTransformer(local_model_path)
            st.success("已加载本地模型")
            return model
        except Exception:
            pass
    # 默认使用轻量级通用模型
    model = SentenceTransformer("all-MiniLM-L6-v2")
    st.success("已加载默认模型 all-MiniLM-L6-v2")
    return model

# ------------------------------
# 3. 图书推荐功能
# ------------------------------
def recommend_books(user_query, df, model, embeddings, top_n=5):
    """根据用户输入推荐相似图书"""
    query_vec = model.encode([user_query])
    similarities = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "书名": row["书名"],
            "作者": row["作者"],
            "作者国籍": row["作者国籍"],
            "简介": row["书本简介"],
            "相似度": float(similarities[idx])
        })
    return results

# ------------------------------
# 4. 聚类展示功能
# ------------------------------
def show_clusters(df):
    if "cluster" not in df.columns or df["cluster"].min() < 0:
        st.warning("当前数据不包含聚类信息，请先运行聚类程序并保存 books_with_clusters.xlsx")
        return
    clusters = sorted(df["cluster"].unique())
    st.subheader("📂 图书聚类（共 {} 个类别）".format(len(clusters)))
    for cid in clusters:
        cluster_books = df[df["cluster"] == cid]
        with st.expander(f"类别 {cid} ｜ 图书数量 {len(cluster_books)}"):
            # 每行展示4本书
            cols = st.columns(4)
            for i, (_, book) in enumerate(cluster_books.iterrows()):
                col = cols[i % 4]
                with col:
                    st.markdown(f"**{book['书名']}**")
                    st.caption(f"{book['作者']} · {book['作者国籍']}")
                    st.text(book["书本简介"][:80] + "...")

# ------------------------------
# 5. AI 图书咨询功能
# ------------------------------
def ai_book_chat(book_name):
    API_KEY = "sk-2e537d80f44b4256bc008000696bae7a"
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.deepseek.com"
    )
    prompt = f"""根据用户输入的图书《{book_name}》，向用户详细介绍这本书的情节、主题、人物、作者及其创作背景等基本信息，并显示对这本书的主流评价。"""
    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "你是一个资深的图书推荐专家，对文学作品有深刻理解。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用 API 失败：{str(e)}"

# ------------------------------
# 6. 主界面：侧边栏导航
# ------------------------------
def main():
    # 加载所有资源
    df = load_data()
    embeddings = load_embeddings()
    model = load_model()
    if embeddings is None:
        st.stop()

    # 侧边栏菜单
    st.sidebar.title("🔍 功能导航")
    menu = st.sidebar.radio(
        "请选择服务",
        ["📖 图书推荐", "📊 图书聚类", "🤖 AI 图书咨询"]
    )

    # ---------- 图书推荐 ----------
    if menu == "📖 图书推荐":
        st.header("📌 根据偏好推荐图书")
        user_input = st.text_area("描述你喜欢的图书类型、情节、主题或关键词", 
                                  "例如：喜欢悬疑推理，或爱情与战争的史诗故事")
        if st.button("🔎 推荐相似图书"):
            if user_input.strip():
                with st.spinner("正在为你寻找好书..."):
                    recs = recommend_books(user_input, df, model, embeddings, top_n=5)
                st.subheader("✨ 推荐结果（按相似度排序）")
                for r in recs:
                    with st.container():
                        st.markdown(f"### 《{r['书名']}》")
                        st.markdown(f"**作者**：{r['作者']} （{r['作者国籍']}）")
                        st.markdown(f"**简介**：{r['简介']}")
                        st.markdown(f"**相似度**：{r['相似度']:.4f}")
                        st.divider()
            else:
                st.warning("请输入阅读偏好")

    # ---------- 图书聚类 ----------
    elif menu == "📊 图书聚类":
        st.header("📚 图书聚类展示（已预分为 5 类）")
        show_clusters(df)

    # ---------- AI 图书咨询 ----------
    else:
        st.header("🤖 AI 图书信息咨询")
        book_list = df["书名"].unique().tolist()
        selected_book = st.selectbox("选择你感兴趣的图书", book_list)
        if st.button("📖 获取详细介绍"):
            with st.spinner("正在向 AI 获取信息..."):
                answer = ai_book_chat(selected_book)
            st.markdown("### 📝 图书详情")
            st.write(answer)

if __name__ == "__main__":
    main()