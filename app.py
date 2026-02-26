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
    page_title="Titanic AI Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium ChatGPT-style UI with Glassmorphism
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

    /* Glassmorphism Containers */
    div[data-testid="stMetricValue"] {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 20px !important;
        border: 1px solid var(--border);
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="stMetricValue"]:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: var(--primary);
        box-shadow: 0 12px 40px 0 rgba(14, 165, 233, 0.2);
    }

    /* Chat Styling */
    [data-testid="stChatMessage"] {
        background: var(--card-bg);
        border-radius: 20px;
        margin-bottom: 20px;
        border: 1px solid var(--border);
        padding: 1.5rem;
        animation: slideIn 0.5s ease-out;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

    /* Spikes Animation */
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
    .spike:nth-child(2) { animation-delay: 0.15s; }
    .spike:nth-child(3) { animation-delay: 0.3s; }
    .spike:nth-child(4) { animation-delay: 0.45s; }
    .spike:nth-child(5) { animation-delay: 0.6s; }

    @keyframes pulse {
        0%, 100% { height: 12px; opacity: 0.4; transform: scaleY(1); }
        50% { height: 28px; opacity: 1; transform: scaleY(1.2); }
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Sidebar Dashboard ----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
    st.title("🗂️ Dataset Intel")
    
    st.info("""
    **Hybrid AI Engine** v2.0
    Combining LLM reasoning with high-performance Pandas execution.
    """)
    
    st.divider()
    st.subheader("💡 Expert Prompts")
    prompts = [
        "Survival rate of 1st class vs 3rd class",
        "Age distribution of survivors",
        "Fare vs Age correlation",
        "Port embarkation count by gender",
        "Percentage of children who survived"
    ]
    for p in prompts:
        if st.button(p, use_container_width=True):
            st.session_state.pushed_prompt = p

    st.divider()
    if st.checkbox("🔍 View Raw Matrix"):
        st.dataframe(pd.read_csv("titanic.csv").head(15), height=300)

# ---------------- Data Core ----------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# ---------------- Main Metrics ----------------
st.title("🚢 Titanic Hybrid AI Assistant")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Total Passengers", len(df))
with m2: st.metric("Survival Rate", f"{(df['Survived'].mean()*100):.1f}%")
with m3: st.metric("Median Age", f"{df['Age'].median():.0f}")
with m4: st.metric("Avg. Fare", f"${df['Fare'].mean():.2f}")

st.markdown("---")

# ---------------- Hybrid Intelligence ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("Missing OpenRouter API Key. Check `.env`.")
    st.stop()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="openai/gpt-4o-mini",
    temperature=0
)

def build_analysis_plan(query):
    """LLM parses query into a structured computational plan."""
    system_prompt = """
    Parse Titanic dataset queries into JSON.
    Columns: Survived (0,1), Pclass (1,2,3), Sex (male,female), Age, Fare, Embarked (C,Q,S).
    Output JSON: {
        "intent": "stat" | "visual",
        "column": "col_name",
        "operation": "mean" | "count" | "correlation" | "distribution",
        "filters": {"col": "val", "col_ops": ">" | "<" | "=="},
        "compare": "col_name" | null
    }
    Example: "Women in class 1 survival rate" -> {"intent": "stat", "column": "Survived", "operation": "mean", "filters": {"Sex": "female", "Pclass": 1}}
    ONLY JSON.
    """
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
        res = llm.invoke(messages).content
        if "```json" in res:
            res = re.search(r'```json\n(.*?)\n```', res, re.DOTALL).group(1)
        return json.loads(res.strip())
    except:
        return None

def execute_plan(plan, df):
    """Pandas executes the computational plan."""
    if not plan: return "I couldn't decode that query. Try something specific like 'Survival rate of men over 30'.", None
    
    try:
        data = df.copy()
        filters = plan.get("filters", {})
        for col, val in filters.items():
            if col in data.columns:
                if isinstance(val, (int, float)):
                    data = data[data[col] == val]
                else:
                    data = data[data[col] == val]

        intent = plan.get("intent")
        col = plan.get("column")
        op = plan.get("operation")
        compare = plan.get("compare")

        answer = ""
        fig = None

        if intent == "stat":
            if op == "mean":
                val = data[col].mean()
                label = "Survival Rate" if col == "Survived" else f"Average {col}"
                answer = f"The **{label}** for this group is **{val*100 if col=='Survived' else val:.2f}{'%' if col=='Survived' else ''}**."
            elif op == "count":
                answer = f"Found **{len(data)}** passengers matching your criteria."

        elif intent == "visual":
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.style.use('dark_background')
            if op == "distribution":
                sns.histplot(data=data, x=col, kde=True, ax=ax, color='#38bdf8', palette="mako")
                answer = f"Showing the distribution of **{col}**."
            elif op == "count" or compare:
                sns.countplot(data=data, x=col, hue=compare, ax=ax, palette="viridis")
                answer = f"Frequency analysis of **{col}**{' compared by ' + compare if compare else ''}."
            
            ax.set_facecolor('none')
            fig.patch.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        return answer if answer else "Analysis complete, but no specific insight found.", fig
    except Exception as e:
        return f"Computation Error: {e}", None

# ---------------- Chat System ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "fig" in msg and msg["fig"]:
            st.pyplot(msg["fig"])

# Handling Prompt Injection from Sidebar/Input
prompt = st.chat_input("Message Titanic AI...")
if "pushed_prompt" in st.session_state:
    prompt = st.session_state.pushed_prompt
    del st.session_state.pushed_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("""
            <div class="loading-box">
                <div class="spike"></div><div class="spike"></div><div class="spike"></div>
                <div class="spike"></div><div class="spike"></div>
                <span style="font-size: 0.9rem; color: #94a3b8; font-family: 'Fira Code';">AI.THINKING()</span>
            </div>
        """, unsafe_allow_html=True)
        
        plan = build_analysis_plan(prompt)
        status.markdown("""
            <div class="loading-box">
                <div class="spike"></div><div class="spike"></div><div class="spike"></div>
                <div class="spike"></div><div class="spike"></div>
                <span style="font-size: 0.9rem; color: #94a3b8; font-family: 'Fira Code';">PANDAS.EXECUTE()</span>
            </div>
        """, unsafe_allow_html=True)
        
        res, fig = execute_plan(plan, df)
        status.empty()
        
        st.markdown(res)
        if fig: st.pyplot(fig)
        st.session_state.messages.append({"role": "assistant", "content": res, "fig": fig})
