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
    page_title="Titanic Intel AI v6.0",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ChatGPT-Style UI with Context Awareness
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

    /* Logic/Memory Trace */
    .memory-trace {
        padding: 8px 12px;
        background: rgba(37, 99, 235, 0.1);
        border-right: 3px solid var(--secondary);
        border-radius: 4px;
        font-family: 'Fira Code', monospace;
        font-size: 0.75rem;
        color: #60a5fa;
        margin-bottom: 10px;
        text-align: right;
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

# ---------------- Initialization ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_plan" not in st.session_state:
    st.session_state.last_plan = None

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
    st.title("🧠 Context Hub")
    st.info("**v6.0 Memory Layer** enabled. Conversational history and pronoun resolution are now active.")
    
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_plan = None
        st.rerun()

    st.divider()
    st.subheader("💡 Analysis Flow")
    st.markdown("""
    1. **Turn 1**: "Who is the first passenger?"
    2. **Turn 2**: "How old were **they**?"
    3. **Turn 3**: "What was **their** fare?"
    """)
    st.divider()
    if st.checkbox("🔍 Dataset Lab"):
        st.dataframe(df.head(10))

# ---------------- Header ----------------
st.title("🚢 Titanic Contextual AI v6.0")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Historical Size", len(df))
with m2: st.metric("Survivors", df['Survived'].sum())
with m3: st.metric("Memory Slots", "Active")
with m4: st.metric("Reasoning", "v6-Turbo")

st.markdown("---")

# ---------------- Intelligence Engine ----------------
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

def build_contextual_plan(query, history, last_plan):
    """LLM derives logic considering previous context and history."""
    
    # Format history for LLM
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
    
    system_prompt = f"""
    You are an expert Titanic Data Scientist with conversational memory.
    Columns: {df.columns.tolist()}
    
    Current History:
    {history_str}
    
    Last Logic Plan Used:
    {json.dumps(last_plan) if last_plan else "None"}
    
    Rules:
    - If user uses pronouns ('they', 'their', 'that group', 'those passengers'), refer to the `Last Logic Plan Used`.
    - Detect 'percentage', 'ratio', 'mean', 'count', 'first_record' as operations.
    - If user asks a new question, ignore old filters unless specified.
    
    Output JSON ONLY:
    {{
        "op": "percentage" | "ratio" | "mean" | "count" | "first_record" | "visual",
        "target": "col_name" | null,
        "filters": [
            {{"col": "Name", "val": value, "comp": "==" | "<" | ">"}}
        ],
        "use_previous_context": true | false,
        "trace": "Reasoning for context resolution"
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

def execute_plan(plan, df, last_plan):
    """Execute plan with context merging."""
    if not plan: return "I couldn't resolve the context of your question.", None
    
    try:
        active_filters = plan.get("filters", [])
        
        # Merge context if requested
        if plan.get("use_previous_context") and last_plan:
            active_filters = last_plan.get("filters", []) + active_filters
        
        # Apply Logic
        working_data = df.copy()
        for f in active_filters:
            col, val, comp = f["col"], f["val"], f["comp"]
            if comp == "==": working_data = working_data[working_data[col] == val]
            elif comp == "<": working_data = working_data[working_data[col] < val]
            elif comp == ">": working_data = working_data[working_data[col] > val]

        if working_data.empty:
            return "Based on that context, no matching records were found.", None

        op = plan.get("op")
        target = plan.get("target")
        trace = plan.get("trace", "")
        
        answer = ""
        fig = None

        if op == "first_record":
            rec = working_data.iloc[0]
            answer = f"**First Passenger in Context:**\n- **Name:** {rec.get('Name', 'Unknown')}\n- **Age:** {rec.get('Age', 'N/A')}\n- **Fare:** ${rec.get('Fare', 0):.2f}\n- **Class:** {rec.get('Pclass')}"
        
        elif op == "percentage":
            val = (len(working_data) / len(df)) * 100
            answer = f"This group represents **{val:.2f}%** of all passengers."
            
        elif op == "mean":
            val = working_data[target].mean()
            if target == "Survived":
                answer = f"The survival rate for this group is **{val*100:.2f}%**."
            else:
                answer = f"The average {target} is **{val:.2f}**."

        elif op == "count":
            answer = f"Found **{len(working_data)}** passengers in this context."

        elif op == "visual":
            fig, ax = plt.subplots(figsize=(10, 4))
            plt.style.use('dark_background')
            sns.histplot(data=working_data, x=target, kde=True, color='#38bdf8', ax=ax)
            ax.set_facecolor('none')
            fig.patch.set_facecolor('none')
            answer = f"Visualizing {target} distribution for current context."

        # Save context
        plan["filters"] = active_filters # Save the merged filters for next turn
        st.session_state.last_plan = plan

        return answer, fig
    except Exception as e:
        return f"Operational Failure: {e}", None

# ---------------- Chat Workflow ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "fig" in m and m["fig"]: st.pyplot(m["fig"])

if prompt := st.chat_input("Continue our conversation about Titanic..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        loading = st.empty()
        loading.markdown("""
            <div class="spike-container">
                <div class="spike"></div><div class="spike"></div><div class="spike"></div>
                <span style="font-size: 0.8rem; color: #94a3b8; font-family: 'Fira Code';">RESOLVING_CONTEXT_MEMORY...</span>
            </div>
        """, unsafe_allow_html=True)
        
        plan = build_contextual_plan(prompt, st.session_state.messages[:-1], st.session_state.last_plan)
        
        if plan and plan.get("use_previous_context"):
            st.markdown(f'<div class="memory-trace">🧠 Context: {plan.get("trace")}</div>', unsafe_allow_html=True)
        
        res, fig = execute_plan(plan, df, st.session_state.last_plan)
        loading.empty()
        
        st.markdown(res)
        if fig: st.pyplot(fig)
        st.session_state.messages.append({"role": "assistant", "content": res, "fig": fig})
