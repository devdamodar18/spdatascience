import streamlit as st
import pandas as pd
from core.engine import SPDSEngine
from core.ai_assistant import SPDSAssistant
from core.learning import LearningProfile

st.set_page_config(page_title="SPDS AI Data Intelligence", page_icon="🧠", layout="wide")

if "df" not in st.session_state: st.session_state.df=None
if "name" not in st.session_state: st.session_state.name=None
if "analysis" not in st.session_state: st.session_state.analysis=None
if "chat" not in st.session_state: st.session_state.chat=[]
if "learning" not in st.session_state: st.session_state.learning=LearningProfile()

st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:1500px}
.hero{padding:25px;border-radius:20px;margin-bottom:18px;background:linear-gradient(135deg,rgba(99,102,241,.16),rgba(14,165,233,.10));border:1px solid rgba(99,102,241,.22)}
</style>""",unsafe_allow_html=True)

with st.sidebar:
    st.title("🧠 SPDS")
    st.caption("AI Data Intelligence Workspace")
    f=st.file_uploader("Upload CSV / Excel",type=["csv","xlsx","xls"])
    if f:
        try:
            if f.name != st.session_state.name:
                st.session_state.df=pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
                st.session_state.name=f.name
                st.session_state.analysis=None
                st.session_state.chat=[]
        except Exception as e: st.error(str(e))
    st.divider()
    page=st.radio("Workspace",["🏠 Home","⚡ Auto Analysis","🤖 Ask SPDS","📊 Visual Studio","🔎 Data Explorer","🧪 Predictive Lab","💡 Insights","🧠 Learning"],label_visibility="collapsed")

df=st.session_state.df
if df is None:
    st.markdown('<div class="hero"><h1>SPDS — AI Data Intelligence</h1><p>Upload your data and let SPDS explain it, visualize it, find patterns and guide your analysis.</p></div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.markdown("### ⚡ Auto Analysis\nOne-click analysis")
    b.markdown("### 🤖 Ask SPDS\nPlain English")
    c.markdown("### 📊 Visual Studio\nInteractive charts")
    d.markdown("### 🧠 Learning\nPersonalized workspace")
    st.info("No CSE background is required.")
    st.stop()

engine=SPDSEngine(df)
assistant=SPDSAssistant(df,engine)
p=engine.profile()

if page=="🏠 Home":
    st.markdown(f'<div class="hero"><h1>Welcome to SPDS</h1><p><b>{st.session_state.name}</b> is ready. Ask questions or run Auto Analysis.</p></div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric("Rows",f"{p['rows']:,}"); b.metric("Columns",f"{p['columns']:,}"); c.metric("Missing",f"{p['missing_cells']:,}"); d.metric("Duplicates",f"{p['duplicates']:,}")
    if st.button("⚡ Run Auto Analysis",type="primary",use_container_width=True):
        with st.spinner("SPDS is analyzing..."):
            st.session_state.analysis=engine.auto_analysis()
            st.session_state.learning.record("analysis","auto")
        st.success("Complete. Open Auto Analysis.")
    st.subheader("Try these")
    for text in ["Explain my data","Find important patterns","What should I do next?"]:
        if st.button(text,use_container_width=True):
            st.session_state.chat += [("user",text),("assistant",assistant.answer(text))]
            st.session_state.learning.record("question",text)
    for role,msg in st.session_state.chat[-8:]:
        with st.chat_message(role): st.write(msg)

elif page=="⚡ Auto Analysis":
    st.title("⚡ Auto Analysis")
    if st.session_state.analysis is None:
        if st.button("Run complete analysis",type="primary"):
            with st.spinner("Profiling, checking quality, finding relationships and selecting visuals..."):
                st.session_state.analysis=engine.auto_analysis()
                st.session_state.learning.record("analysis","auto")
        else:
            st.info("Click the button to let SPDS decide which analyses are useful."); st.stop()
    r=st.session_state.analysis
    for x in r["summary"]: st.markdown("- "+x)
    st.subheader("Recommended visualizations")
    for spec in r["charts"]:
        st.plotly_chart(assistant.make_chart(spec),use_container_width=True)
    st.subheader("Data quality")
    st.dataframe(r["quality"],use_container_width=True,hide_index=True)
    st.subheader("Relationships")
    st.dataframe(r["correlations"],use_container_width=True,hide_index=True)

elif page=="🤖 Ask SPDS":
    st.title("🤖 Ask SPDS")
    st.caption("Use normal language. You do not need programming knowledge.")
    for q in ["Explain this dataset like I'm a beginner","What are the most important patterns?","Are there data quality problems?","Which charts should I use?","Find unusual values","What should I analyze next?"]:
        if st.button(q,use_container_width=True):
            st.session_state.chat += [("user",q),("assistant",assistant.answer(q))]
            st.session_state.learning.record("question",q)
    q=st.chat_input("Ask SPDS anything about your data...")
    if q:
        st.session_state.chat += [("user",q),("assistant",assistant.answer(q))]
        st.session_state.learning.record("question",q)
    for role,msg in st.session_state.chat[-16:]:
        with st.chat_message(role): st.write(msg)
    st.caption("This version is local-first. An optional LLM connector can be added later for richer reasoning.")

elif page=="📊 Visual Studio":
    st.title("📊 Visual Studio")
    opts=assistant.chart_options()
    if not opts: st.info("Not enough suitable fields.")
    else:
        labels=[x["label"] for x in opts]
        s=opts[labels.index(st.selectbox("Choose a recommended visualization",labels))]
        st.plotly_chart(assistant.make_chart(s),use_container_width=True)
        st.info(s["reason"])

elif page=="🔎 Data Explorer":
    st.title("🔎 Data Explorer")
    term=st.text_input("Search columns")
    cols=[c for c in df.columns if term.lower() in c.lower()] if term else list(df.columns)
    st.dataframe(df[cols].head(2000),use_container_width=True,height=560,hide_index=True)
    st.download_button("Download visible data",df[cols].to_csv(index=False),"spds_export.csv","text/csv")

elif page=="🧪 Predictive Lab":
    st.title("🧪 Predictive Lab")
    target=st.selectbox("What would you like to predict?",list(df.columns))
    if st.button("Run prediction",type="primary"):
        with st.spinner("Training baseline model..."):
            r=engine.predict(target)
        if r["ok"]:
            a,b=st.columns(2); a.metric("Model",r["model"]); b.metric("Validation",r["score"])
            st.info(r["explanation"]); st.dataframe(r["importance"],use_container_width=True,hide_index=True)
        else: st.warning(r["message"])

elif page=="💡 Insights":
    st.title("💡 Insights")
    for x in engine.findings(): st.markdown("- "+x)
    st.subheader("Next analyses")
    for x in engine.next_steps(): st.markdown("- "+x)
    st.subheader("Anomaly candidates")
    st.dataframe(engine.anomalies(),use_container_width=True,hide_index=True)

else:
    st.title("🧠 Learning & Personalization")
    s=st.session_state.learning.summary()
    a,b,c=st.columns(3); a.metric("Analyses",s["analyses"]); b.metric("Questions",s["questions"]); c.metric("Actions",s["actions"])
    st.info("SPDS learns usage preferences safely. It does not silently rewrite its own production code.")
    if st.button("Reset session learning"):
        st.session_state.learning=LearningProfile(); st.rerun()
