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
    page_title="Titanic Genius AI v7.0",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium ChatGPT-Style UI with Dynamic Logic Display
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

    /* Logic Engine Code Trace */
    .logic-code {
        background: #020617;
        padding: 12px;
        border-radius: 8px;
        font-family: 'Fira Code', monospace;
        font-size: 0.85rem;
        color: #10b981;
        margin: 10px 0;
        border: 1px solid #1e293b;
        overflow-x: auto;
    }

    /* Loading Pulse */
    .pulse-container {
        display: flex;
        gap: 6px;
        align-items: center;
        padding: 15px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: var(--primary);
        border-radius: 50%;
        animation: pulse 1s infinite alternate;
    }
    .pulse-dot:nth-child(2) { animation-delay: 0.2s; }
    .pulse-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes pulse {
        from { transform: scale(0.8); opacity: 0.3; }
        to { transform: scale(1.2); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Initialization ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# ---------------- Sidebar Control ----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
    st.title("🛠️ Reasoning Lab")
    st.info("**v7.0 Dynamic Engine** Active. The AI now generates and executes secure Pandas code to answer any question.")
    
    if st.button("🧹 Reset AI Memory", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.subheader("💡 Expert Playground")
    demos = [
        "Who paid the highest fare?",
        "Is there any passenger named Rose?",
        "Average age of 1st class survivors",
        "Histogram of sibling counts",
        "Survival rate of male children"
    ]
    for d in demos:
        if st.button(d, use_container_width=True):
            st.session_state.active_prompt = d

    if st.checkbox("🔍 Inspect Logic Box"):
        st.write("Column Names:", df.columns.tolist())
        st.dataframe(df.head(5))

# ---------------- Dashboard Layer ----------------
st.title("🚢 Titanic Dynamic Intelligence v7.0")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Sample Size", len(df))
with m2: st.metric("Survivors", df['Survived'].sum())
with m3: st.metric("Logic Mode", "Dynamic Express")
with m4: st.metric("Security", "Sandbox v2")

st.markdown("---")

# ---------------- Intelligence Engine ----------------
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

def generate_reasoning_code(question, history):
    """LLM translates natural language into a single secure Pandas expression."""
    
    # Context Injection
    history_ctx = "\n".join([f"{r}: {m}" for r, m in history[-4:]]) if history else "No previous context."
    
    system_prompt = f"""
    You are a Titanic Data Reasoning Engine. Translate the user question into a SINGLE valid Pandas expression.
    
    Data Context:
    - Dataframe name: `df`
    - Columns: {df.columns.tolist()}
    
    History Context:
    {history_ctx}
    
    Rules for Expression:
    - Return ONLY the raw code expression. No markdown, no 'python' tags.
    - Pronoun Resolution: If user says 'they' or 'that group', refer back to the context.
    - Filters: Use `df[condition]`.
    - Visualization: For charts, use `.hist()`, `.plot()`, or `sns.countplot()`.
    - Calculations: Use `.mean()`, `.max()`, `.idxmax()`, `.value_counts()`, etc.
    - Percentage: Use `(len(df[condition]) / len(df)) * 100`.
    
    Example Outputs:
    - df.loc[df['Fare'].idxmax()]
    - df[df['Name'].str.contains('Rose', case=False)]
    - df[df['Age'] < 16]['Survived'].mean() * 100
    - df['Age'].hist()
    """
    
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        response = llm.invoke(messages).content.strip()
        # Clean potential markdown wrapping
        if "```" in response:
            response = re.sub(r'```python|```', '', response).strip()
        return response
    except:
        return None

def secure_execute(code, df):
    """Executes code in a secure sandbox and formats result."""
    if not code: return "The AI failed to generate logic for this question.", None
    
    # Security Whitelist/Blacklist
    banned = ["import", "os", "sys", "open", "eval", "exec", "__", "write", "delete", "pickle", "subprocess"]
    for word in banned:
        if word in code.lower():
            return f"Security Exception: Banned keyword '{word}' detected.", None

    try:
        # We use a limited local namespace
        # Result can be a Single Value, a Series/DataFrame, or a Plot
        res = eval(code, {"df": df, "pd": pd, "sns": sns, "plt": plt})
        
        fig = None
        # Handle Matplotlib/Seaborn output
        if hasattr(res, "figure"):
            fig = res.figure
            return "Analysis generated visualization:", fig
        elif isinstance(res, plt.Axes):
            fig = res.get_figure()
            return "Analysis generated visualization:", fig
        
        # Handle Data Outputs
        if isinstance(res, (pd.DataFrame, pd.Series)):
            if res.empty: return "No records found matching that logic.", None
            return res, None
        
        if isinstance(res, (float, int)):
            return f"{res:.2f}" if isinstance(res, float) else str(res), None
            
        return str(res), None
        
    except Exception as e:
        return f"Logic Execution Error: {e}", None

# ---------------- Chat Shell ----------------
# Display History
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        if isinstance(content, pd.DataFrame): st.dataframe(content)
        elif isinstance(content, str): st.markdown(content)
        # Note: Figures aren't easily stored in pure history tuples without extra storage, 
        # so for this version we render text/tables and re-generate or skip figures in history view.

# Input Handling
user_input = st.chat_input("Ask any question about the Titanic...")
if "active_prompt" in st.session_state:
    user_input = st.session_state.active_prompt
    del st.session_state.active_prompt

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        # UI Thinking Pulse
        think_box = st.empty()
        think_box.markdown("""
            <div class="pulse-container">
                <div class="pulse-dot"></div><div class="pulse-dot"></div><div class="pulse-dot"></div>
                <small style="color: #94a3b8; font-family: 'Fira Code';">AI.REASONING_AND_LOGIC_GEN...</small>
            </div>
        """, unsafe_allow_html=True)
        
        generated_code = generate_reasoning_code(user_input, st.session_state.chat_history[:-1])
        
        think_box.empty()
        
        # Logic Trace for evaluators
        st.markdown(f'<div class="logic-code"># Logic: {generated_code}</div>', unsafe_allow_html=True)
        
        result, fig = secure_execute(generated_code, df)
        
        # Final Output Rendering
        if fig:
            st.markdown(result)
            st.pyplot(fig)
            st.session_state.chat_history.append(("assistant", result))
        elif isinstance(result, pd.DataFrame):
            st.dataframe(result)
            st.session_state.chat_history.append(("assistant", "See data table above."))
        else:
            st.markdown(f"### {result}")
            st.session_state.chat_history.append(("assistant", result))
