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
    page_title="Titanic Intel AI v4.0",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ChatGPT-Style Professional UI
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
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* Chat Styling */
    [data-testid="stChatMessage"] {
        background: var(--card-bg);
        border-radius: 20px;
        margin-bottom: 20px;
        border: 1px solid var(--border);
        padding: 1.5rem;
    }

    /* Thinking Animation */
    .think-container {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 12px;
        background: rgba(56, 189, 248, 0.05);
        border-radius: 12px;
        border: 1px dashed var(--primary);
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

# ---------------- Data Processing ----------------
@st.cache_data
def load_data():
    # Load local CSV
    df = pd.read_csv("titanic.csv")
    return df

df = load_data()

# ---------------- Sidebar Intel ----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
    st.title("🧊 AI Lab")
    st.info("**v4.0 Advanced Hybrid** layer online. This engine uses deep schema discovery to eliminate NaN and fallback errors.")
    
    st.divider()
    st.subheader("💡 Analysis Presets")
    presets = [
        "Percentage of children who survived",
        "Average age of 1st class females",
        "Survival of males vs females in 3rd class",
        "Fare distribution for C harbor",
        "Total families on board"
    ]
    for p in presets:
        if st.button(p, use_container_width=True):
            st.session_state.active_prompt = p

    if st.checkbox("🔍 Data Matrix"):
        st.dataframe(df.head(10), height=300)

# ---------------- Dashboard Header ----------------
st.title("🚢 Titanic AI Intelligence v4.0")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Passengers", len(df))
with col2: st.metric("Survivors", df['Survived'].sum())
with col3: st.metric("Median Age", f"{df['Age'].median():.0f}")
with col4: st.metric("Avg. Fare", f"${df['Fare'].mean():.2f}")

st.markdown("---")

# ---------------- Intelligence Architecture ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("Missing OpenRouter API Key. Add it to .env.")
    st.stop()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="openai/gpt-4o-mini",
    temperature=0
)

def build_intel_plan(query, df):
    """Deep Schema Discovery & Intent Detection."""
    schema = {
        "columns": df.columns.tolist(),
        "sex_vals": ["male", "female"],
        "embarked_vals": ["S", "C", "Q"],
        "pclass_vals": [1, 2, 3]
    }
    
    system_prompt = f"""
    You are an expert data engineer. Translate Titanic queries into a precise JSON analysis plan.
    Schema: {json.dumps(schema)}
    
    Rules for Precision:
    - 'children' -> Filter `Age < 16`.
    - 'survival rate' -> Mean of `Survived`.
    - 'percentage' -> (Subset Count / Total Count) * 100.
    - 'class' -> Use `Pclass` (1, 2, or 3).
    - 'embarked' -> Values are 'S' (Southampton), 'C' (Cherbourg), 'Q' (Queenstown).

    Output JSON ONLY:
    {{
        "intent": "stat" | "visual",
        "target": "col_name",
        "op": "mean" | "count" | "sum",
        "filters": [
            {{"col": "Sex", "val": "female", "comp": "=="}},
            {{"col": "Age", "val": 16, "comp": "<"}}
        ],
        "compare_col": "col_name" | null,
        "thought": "Briefly explain logic."
    }}
    """
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
        raw_res = llm.invoke(messages).content
        if "```json" in raw_res:
            raw_res = re.search(r'```json\n(.*?)\n```', raw_res, re.DOTALL).group(1)
        return json.loads(raw_res.strip())
    except:
        return None

def run_it(plan, df):
    """Fail-safe execution engine."""
    if not plan: return "I couldn't decode that query. Try saying 'Survival rate of females'.", None
    
    try:
        data = df.copy()
        
        # Robust filtering
        for f in plan.get("filters", []):
            col, val, comp = f["col"], f["val"], f["comp"]
            if comp == "==": data = data[data[col] == val]
            elif comp == "<": data = data[data[col] < val]
            elif comp == ">": data = data[data[col] > val]

        if data.empty:
            return "Based on your filters, no matching records were found in the dataset.", None

        intent = plan.get("intent")
        target = plan.get("target")
        op = plan.get("op")
        compare = plan.get("compare_col")
        
        answer = ""
        fig = None

        if intent == "stat":
            if op == "count":
                answer = f"Found **{len(data)}** passengers matching your criteria."
            elif op == "mean":
                val = data[target].mean()
                if target == "Survived":
                    answer = f"The survival rate for this selection is **{val*100:.2f}%**."
                else:
                    answer = f"The average **{target}** is **{val:.2f}**."
            elif op == "sum":
                answer = f"The total sum of **{target}** is **{data[target].sum():.0f}**."

        elif intent == "visual":
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.style.use('dark_background')
            if compare:
                sns.countplot(data=data, x=target, hue=compare, palette="mako", ax=ax)
                answer = f"Comparison chart of **{target}** by **{compare}**."
            else:
                sns.histplot(data=data, x=target, kde=True, color='#38bdf8', ax=ax)
                answer = f"Distribution of **{target}** displayed below."
            
            # Styling
            ax.set_facecolor('none')
            fig.patch.set_facecolor('none')
            ax.grid(False)

        return answer, fig
    except Exception as e:
        return f"Operational Error: {e}", None

# ---------------- Chat Shell ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# View History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "fig" in m and m["fig"]: st.pyplot(m["fig"])

# Handling Inputs
prompt = st.chat_input("Enter your research query...")
if "active_prompt" in st.session_state:
    prompt = st.session_state.active_prompt
    del st.session_state.active_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        # Display Thinking Phase
        think_placeholder = st.empty()
        think_placeholder.markdown("""
            <div class="think-container">
                <div class="spike"></div><div class="spike"></div><div class="spike"></div>
                <div class="spike"></div><div class="spike"></div>
                <span style="font-family: 'Fira Code'; font-size: 0.85rem; color: #94a3b8;">DISCOVERING_SCHEMA...</span>
            </div>
        """, unsafe_allow_html=True)
        
        plan = build_intel_plan(prompt, df)
        
        think_placeholder.markdown(f"""
            <div class="think-container">
                <div class="spike"></div><div class="spike"></div><div class="spike"></div>
                <div class="spike"></div><div class="spike"></div>
                <span style="font-family: 'Fira Code'; font-size: 0.85rem; color: #94a3b8;">PLAN: {plan.get('thought', 'Executing compute')}</span>
            </div>
        """, unsafe_allow_html=True)
        
        res, fig = run_it(plan, df)
        think_placeholder.empty()
        
        st.markdown(res)
        if fig: st.pyplot(fig)
        st.session_state.messages.append({"role": "assistant", "content": res, "fig": fig})
