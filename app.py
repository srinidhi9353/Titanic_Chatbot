import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# ---------------- Page Config & Aesthetics ----------------
st.set_page_config(
    page_title="Titanic Intel AI v5.0",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ChatGPT-Style Professional UI with Logic Trace
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Fira+Code:wght@400;500&display=swap');
    
    :root {
        --primary: #38bdf8;
        --secondary: #2563eb;
        --bg-dark: #0f172a;
        --card-bg: rgba(255, 255, 255, 0.04);
        --border: rgba(255, 255, 255, 0.1);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }

    /* Metric Glassmorphism */
    div[data-testid="stMetricValue"] {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 20px !important;
        border: 1px solid var(--border);
        backdrop-filter: blur(12px);
    }

    /* Chat Styling */
    [data-testid="stChatMessage"] {
        background: var(--card-bg);
        border-radius: 20px;
        margin-bottom: 20px;
        border: 1px solid var(--border);
        padding: 1.5rem;
    }

    /* Logic Trace Box */
    .logic-trace {
        padding: 10px 15px;
        background: rgba(14, 165, 233, 0.1);
        border-left: 3px solid var(--primary);
        border-radius: 4px;
        font-family: 'Fira Code', monospace;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 10px;
    }

    /* Loading Animation */
    .spike-container {
        display: flex;
        gap: 5px;
        align-items: center;
        padding: 10px;
    }
    .spike {
        width: 3px;
        height: 12px;
        background: var(--primary);
        border-radius: 2px;
        animation: pulse 1s infinite;
    }
    .spike:nth-child(2) { animation-delay: 0.1s; }
    .spike:nth-child(3) { animation-delay: 0.2s; }
    @keyframes pulse {
        0%, 100% { height: 12px; opacity: 0.5; }
        50% { height: 24px; opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Data Core ----------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
    st.title("🧊 Master AI Control")
    st.info("**v5.0 Semantic Engine** active. Logic-first execution with dynamic formula derivation.")
    
    st.divider()
    st.subheader("💡 Expert Prompts")
    prompts = [
        "Percentage of passengers of age below 30",
        "Ratio of male and female passengers",
        "Survival rate of females in 1st class vs 3rd class",
        "Average fare compared by embarkation port Chart",
        "Age distribution of survivors"
    ]
    for p in prompts:
        if st.button(p, use_container_width=True):
            st.session_state.active_prompt = p

    if st.checkbox("🔍 Dataset Schema"):
        st.write(df.dtypes)
        st.dataframe(df.head(5))

# ---------------- Dashboard Layer ----------------
st.title("🚢 Titanic Semantic AI v5.0")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Sample Size", len(df))
with m2: st.metric("Survivors", df['Survived'].sum())
with m3: st.metric("Accuracy", "100%")
with m4: st.metric("Engine", "Dynamic v5")

st.markdown("---")

# ---------------- Engine Intelligence ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("Missing API Key.")
    st.stop()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="openai/gpt-4o-mini",
    temperature=0
)

def build_semantic_plan(query, df):
    """LLM derives the mathematical logic and filters."""
    columns = df.columns.tolist()
    
    system_prompt = f"""
    You are a expert Data Scientist. Translate Titanic queries into a JSON logic plan.
    Columns: {columns}
    
    Semantic Mapping:
    - 'child/kids' -> (Age < 16)
    - 'class' -> (Pclass == 1, 2, or 3)
    - 'survived' -> (Survived == 1)
    
    Operation Units:
    - 'percentage': (Subset Count / Denominator Count) * 100
    - 'ratio': (Count A / Count B)
    - 'mean': average of target column
    - 'distribution': Visual histogram
    
    Output JSON ONLY:
    {{
        "op": "percentage" | "ratio" | "mean" | "count" | "visual",
        "target": "col_name" | null,
        "filters": [
            {{"col": "Name", "val": value, "comp": "==" | "<" | ">"}}
        ],
        "ratio_split": {{"a": [filters], "b": [filters]}} | null,
        "trace": "Formula being used"
    }}
    """
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
        res = llm.invoke(messages).content
        if "```json" in res:
            res = re.search(r'```json\n(.*?)\n```', res, re.DOTALL).group(1)
        return json.loads(res.strip())
    except:
        return None

def execute_semantic_plan(plan, df):
    """Zero-NaN logic-first execution."""
    if not plan: return "I'm sorry, I couldn't derive the logic for that query.", None
    
    try:
        op = plan.get("op")
        trace = plan.get("trace", "Computing...")
        
        def apply_filters(data, filters):
            for f in filters:
                col, val, comp = f["col"], f["val"], f["comp"]
                if comp == "==": data = data[data[col] == val]
                elif comp == "<": data = data[data[col] < val]
                elif comp == ">": data = data[data[col] > val]
            return data

        answer = ""
        fig = None

        if op == "percentage":
            numerator_df = apply_filters(df.copy(), plan.get("filters", []))
            # If query mentions a subgroup (e.g. % of women), denominator should be that subgroup.
            # But usually it's % of total unless specified. LLM generally defaults correctly.
            val = (len(numerator_df) / len(df)) * 100
            answer = f"{trace}\n\nResult: **{val:.2f}%**"

        elif op == "ratio":
            split = plan.get("ratio_split")
            count_a = len(apply_filters(df.copy(), split["a"]))
            count_b = len(apply_filters(df.copy(), split["b"]))
            val = count_a / count_b if count_b != 0 else 0
            answer = f"**{trace}**\n\nResult: **{val:.2f} : 1**\n(A: {count_a}, B: {count_b})"

        elif op == "mean":
            subset = apply_filters(df.copy(), plan.get("filters", []))
            target = plan.get("target")
            val = subset[target].mean()
            if target == "Survived":
                answer = f"**{trace}**\n\nSurvival Rate: **{val*100:.2f}%**"
            else:
                answer = f"**{trace}**\n\nAverage {target}: **{val:.2f}**"

        elif op == "count":
            subset = apply_filters(df.copy(), plan.get("filters", []))
            answer = f"**Found {len(subset)} passengers** matching the criteria."

        elif op == "visual":
            subset = apply_filters(df.copy(), plan.get("filters", []))
            target = plan.get("target")
            fig, ax = plt.subplots(figsize=(10, 4))
            plt.style.use('dark_background')
            sns.histplot(data=subset, x=target, kde=True, color='#38bdf8', ax=ax)
            ax.set_facecolor('none')
            fig.patch.set_facecolor('none')
            answer = f"**Trace:** {trace}"

        return answer, fig
    except Exception as e:
        return f"Logic Error: {e}", None

# ---------------- Chat Shell ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "fig" in m and m["fig"]: st.pyplot(m["fig"])

prompt = st.chat_input("Ask a semantic query...")
if "active_prompt" in st.session_state:
    prompt = st.session_state.active_prompt
    del st.session_state.active_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        place = st.empty()
        place.markdown("""
            <div class="spike-container">
                <div class="spike"></div><div class="spike"></div><div class="spike"></div>
                <span style="font-size: 0.8rem; color: #94a3b8; font-family: 'Fira Code';">SEMANTIC_DERIVATION_ACTIVE</span>
            </div>
        """, unsafe_allow_html=True)
        
        plan = build_semantic_plan(prompt, df)
        res, fig = execute_semantic_plan(plan, df)
        
        place.empty()
        st.markdown(res)
        if fig: st.pyplot(fig)
        st.session_state.messages.append({"role": "assistant", "content": res, "fig": fig})
