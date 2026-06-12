import streamlit as st
import pandas as pd
import plotly.express as px
from googleapiclient.discovery import build
from textblob import TextBlob
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard YouTube", page_icon="🎬", layout="wide")

st.title("🎬 Dashboard de Análise de Sentimentos — YouTube")
st.markdown("Analise os comentários de qualquer vídeo público do YouTube em tempo real.")

with st.sidebar:
    st.header("⚙️ Configurações")
    import os
api_key = os.environ.get("YOUTUBE_API_KEY", "")
if not api_key:
    api_key = st.text_input("🔑 Sua API Key do YouTube", type="password")
    video_url = st.text_input("🔗 URL ou ID do vídeo")
    max_comments = st.selectbox("💬 Nº de comentários", [50, 100, 200, 500])
    rodar = st.button("▶️ Analisar vídeo")

def extrair_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url.strip()

def coletar_comentarios(video_id, api_key, max_comments=100):
    youtube = build("youtube", "v3", developerKey=api_key)
    comentarios = []
    next_page_token = None
    video_info = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    titulo = video_info["items"][0]["snippet"]["title"]
    canal = video_info["items"][0]["snippet"]["channelTitle"]
    while len(comentarios) < max_comments:
        resposta = youtube.commentThreads().list(
            part="snippet", videoId=video_id,
            maxResults=min(100, max_comments - len(comentarios)),
            pageToken=next_page_token, textFormat="plainText"
        ).execute()
        for item in resposta["items"]:
            c = item["snippet"]["topLevelComment"]["snippet"]
            comentarios.append({
                "text": c["textDisplay"], "author": c["authorDisplayName"],
                "published_at": c["publishedAt"], "like_count": c["likeCount"],
                "video_title": titulo, "canal": canal
            })
        next_page_token = resposta.get("nextPageToken")
        if not next_page_token:
            break
    df = pd.DataFrame(comentarios)
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["day_of_week"] = df["published_at"].dt.day_name()
    df["hour"] = df["published_at"].dt.hour
    return df

def analisar_sentimento(texto):
    try:
        texto_en = GoogleTranslator(source="auto", target="en").translate(texto[:500])
        return TextBlob(texto_en).sentiment.polarity
    except:
        return 0.0

def classificar(score):
    if score > 0.1: return "Positivo"
    elif score < -0.1: return "Negativo"
    else: return "Neutro"

if rodar and api_key and video_url:
    video_id = extrair_id(video_url)
    with st.spinner("⏳ Coletando e analisando comentários..."):
        df = coletar_comentarios(video_id, api_key, max_comments)
        df["sentiment_score"] = df["text"].apply(analisar_sentimento)
        df["sentiment_label"] = df["sentiment_score"].apply(classificar)

    st.success(f"✅ {len(df)} comentários analisados!")
    st.subheader(f"🎬 {df['video_title'].iloc[0]} — {df['canal'].iloc[0]}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💬 Comentários", len(df))
    col2.metric("😊 Positivos", f"{round(len(df[df.sentiment_label=='Positivo'])/len(df)*100,1)}%")
    col3.metric("😐 Neutros", f"{round(len(df[df.sentiment_label=='Neutro'])/len(df)*100,1)}%")
    col4.metric("😠 Negativos", f"{round(len(df[df.sentiment_label=='Negativo'])/len(df)*100,1)}%")

    st.sidebar.markdown("---")
    filtro = st.sidebar.selectbox("Filtrar por sentimento", ["Todos","Positivo","Neutro","Negativo"])
    df_f = df if filtro == "Todos" else df[df.sentiment_label == filtro]

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.pie(df_f, names="sentiment_label", title="Distribuição de Sentimentos",
            color="sentiment_label", hole=0.4,
            color_discrete_map={"Positivo":"#2ecc71","Neutro":"#95a5a6","Negativo":"#e74c3c"})
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        ordem = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        nomes = {"Monday":"Seg","Tuesday":"Ter","Wednesday":"Qua","Thursday":"Qui",
                 "Friday":"Sex","Saturday":"Sáb","Sunday":"Dom"}
        cnt = df_f.groupby("day_of_week").size().reindex(ordem).reset_index()
        cnt.columns = ["day","total"]
        cnt["day_pt"] = cnt["day"].map(nomes)
        fig2 = px.bar(cnt, x="day_pt", y="total", title="Comentários por Dia da Semana",
            color="total", color_continuous_scale="blues",
            labels={"day_pt":"Dia","total":"Comentários"})
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        hm = df_f.groupby(["day_of_week","hour"]).size().unstack(fill_value=0).reindex(ordem)
        fig3 = px.imshow(hm, title="Heatmap Hora x Dia",
            labels=dict(x="Hora", y="Dia", color="Comentários"),
            color_continuous_scale="YlOrRd", aspect="auto")
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        df_t = df_f.set_index("published_at").resample("D")["sentiment_score"].mean().reset_index()
        fig4 = px.line(df_t, x="published_at", y="sentiment_score",
            title="Evolução do Sentimento", markers=True,
            labels={"published_at":"Data","sentiment_score":"Score"})
        fig4.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("☁️ Nuvem de Palavras")
    texto = " ".join(df_f["text"].tolist())
    wc = WordCloud(width=800, height=400, background_color="white",
        colormap="Blues", max_words=100).generate(texto)
    fig5, ax = plt.subplots(figsize=(14,6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig5)

    st.subheader("🏆 Top Comentários por Curtidas")
    st.dataframe(df_f[["author","text","like_count","sentiment_label","sentiment_score"]]
        .sort_values("like_count", ascending=False).head(10).reset_index(drop=True))

    st.subheader("💡 Insights Automáticos")
    score_medio = round(df["sentiment_score"].mean(), 3)
    hora_pico = df.groupby("hour").size().idxmax()
    dia_pico = df.groupby("day_of_week").size().idxmax()
    mais_curtido = df.loc[df["like_count"].idxmax(), "text"]
    st.info(f"📈 Score médio de sentimento: **{score_medio}**")
    st.info(f"⏰ Horário de maior engajamento: **{hora_pico}h**")
    st.info(f"📅 Dia de maior engajamento: **{dia_pico}**")
    st.info(f"❤️ Comentário mais curtido: *{mais_curtido[:150]}*")

elif rodar:
    st.warning("⚠️ Preencha a API Key e a URL do vídeo!")
