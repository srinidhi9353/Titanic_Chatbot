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
    page_title="Titanic Intelligence AI",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium ChatGPT-style UI
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

    /* Glassmorphism Metrics */
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

    /* AI Thinking Spikes */
    .loading-box {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 15px;
        background: var(--card-bg);
        border-radius: 12px;
        width: fit-content;
    }
    .spike {
        width: 3px;
        height: 12px;
        background: var(--primary);
        border-radius: 2px;
        animation: pulse 1s infinite ease-in-out;
    }
    .spike:nth-child(2) { animation-delay: 0.1s; }
    .spike:nth-child(3) { animation-delay: 0.2s; }
    .spike:nth-child(4) { animation-delay: 0.3s; }
    .spike:nth-child(5) { animation-delay: 0.4s; }

    @keyframes pulse {
        0%, 100% { height: 12px; opacity: 0.4; }
        50% { height: 28px; opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Data Core ----------------
@st.cache_data
def load_data():
    # Use seaborn dataset for variety, but ensure it matches your requirements
    return sns.load_dataset("titanic")

df = load_data()

# ---------------- Sidebar Dashboard ----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
    st.title("🗂️ Intelligence Hub")
    
    st.info("**v3.0 Auto-Detect Engine** active. Now with schema awareness and robust error handling.")
    
    st.divider()
    st.subheader("💡 Expert Prompts")
    prompts = [
        "Percentage of children who survived",
        "Survival rate of 1st class vs 3rd class",
        "Average fare paid by survivors vs non-survivors",
        "Embarkation count by gender chart",
        "Age distribution of passengers"
    ]
    for p in prompts:
        if st.button(p, use_container_width=True):
            st.session_state.active_prompt = p

    if st.checkbox("🔍 Inspect Data Matrix"):
        st.dataframe(df.head(10), height=300)

# ---------------- Header Metrics ----------------
st.title("🚢 Titanic AI Intelligence v3.0")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Survivors", df['survived'].sum())
with m2: st.metric("Overall Survival", f"{(df['survived'].mean()*100):.1f}%")
with m3: st.metric("Median Age", f"{df['age'].median():.0f}")
with m4: st.metric("Max Fare", f"${df['fare'].max():.2f}")

st.markdown("---")

# ---------------- Intelligence Engine ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("Missing OpenRouter API Key.")
    st.stop()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="openai/gpt-4o-mini",
    temperature=0
)

def build_computational_plan(query, df):
    """LLM uses schema awareness to build a precise plan."""
    schema_info = df.dtypes.to_string()
    sample_data = df.head(3).to_string()
    
    system_prompt = f"""
    You are a data engineer for the Titanic dataset.
    Columns: {df.columns.tolist()}
    Dtypes: {schema_info}
    Sample: {sample_data}

    Rules:
    - If user asks for "children", filter by 'age' < 18 or 'who' == 'child'.
    - If user asks for "percentage", calculate (subset_count / total_count) * 100.
    - If user asks for "survival rate", it's the mean of 'survived'.
    - If querying 'class', values are 'First', 'Second', 'Third' (NOT 1, 2, 3).

    Output JSON ONLY:
    {{
        "intent": "stat" | "visual",
        "target_col": "col_name",
        "operation": "mean" | "count" | "percentage" | "sum",
        "filters": [
            {{"col": "name", "op": "==", "val": "value"}},
            {{"col": "age", "op": "<", "val": 18}}
        ],
        "groupby": "col_name" | null,
        "explanation": "Brief reasoning"
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

def execute_computational_plan(plan, df):
    """Executes the plan with robust error handling."""
    if not plan: return "I'm sorry, I couldn't interpret that query. Could you try rephrasing?", None
    
    try:
        data = df.copy()
        filters = plan.get("filters", [])
        
        # Apply filters
        for f in filters:
            col, op, val = f["col"], f["op"], f["val"]
            if op == "==": data = data[data[col] == val]
            elif op == "<": data = data[data[col] < val]
            elif op == ">": data = data[data[col] > val]

        if data.empty:
            return "Based on the criteria, no matching passengers were found.", None

        intent = plan.get("intent")
        target = plan.get("target_col")
        op = plan.get("operation")
        groupby = plan.get("groupby")
        
        answer = ""
        fig = None

        if intent == "stat":
            if op == "percentage":
                val = (len(data) / len(df)) * 100
                answer = f"The percentage of passengers matching your query is **{val:.2f}%**."
            elif op == "mean":
                val = data[target].mean()
                if target == "survived":
                    answer = f"The survival rate for this group was **{val*100:.2f}%**."
                else:
                    answer = f"The average {target} is **{val:.2f}**."
            elif op == "count":
                answer = f"Found **{len(data)}** passengers matching your query."

        elif intent == "visual":
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.style.use('dark_background')
            if groupby:
                sns.countplot(data=data, x=groupby, hue=target if target != groupby else None, ax=ax, palette="mako")
                answer = f"Visualizing counts by **{groupby}**."
            else:
                sns.histplot(data=data, x=target, kde=True, ax=ax, color='#38bdf8')
                answer = f"Distribution of **{target}** displayed."
            
            # Stylize plot
            ax.set_facecolor('none')
            fig.patch.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        return answer, fig
    except Exception as e:
        return f"Operational Error: {e}", None

# ---------------- Chat System ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "fig" in m and m["fig"]: st.pyplot(m["fig"])

# Handling prompt
prompt = st.chat_input("Ask a question about the Titanic...")
if "active_prompt" in st.session_state:
    prompt = st.session_state.active_prompt
    del st.session_state.active_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        st.caption(f"🧠 Reasoning: Parsing intent for '{prompt}'...")
        
        status = st.empty()
        status.markdown("""
            <div class="loading-box">
                <div class="spike"></div><div class="spike"></div><div class="spike"></div>
                <div class="spike"></div><div class="spike"></div>
                <small style="color: #94a3b8; font-family: 'Fira Code';">ENGAGING_CORE_ENGINE</small>
            </div>
        """, unsafe_allow_html=True)
        
        plan = build_computational_plan(prompt, df)
        res, fig = execute_computational_plan(plan, df)
        
        status.empty()
        st.markdown(res)
        if fig: st.pyplot(fig)
        
        st.session_state.messages.append({"role": "assistant", "content": res, "fig": fig})
