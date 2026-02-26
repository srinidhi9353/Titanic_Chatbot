import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# ---------------- Page Config & Aesthetics ----------------
st.set_page_config(
    page_title="Titanic Copilot v7.5",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional ChatGPT-Style UI
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

# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
    st.title("🚢 Data Copilot v7.5")
    st.info("**Conversational Reasoning** active. Pronoun resolution and regex-secure execution enabled.")
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.subheader("💡 Expert Scenarios")
    demos = [
        "Who is aged more?",
        "What is the maximum age?",
        "Ratio of adults and children",
        "Is there a passenger named Rose?",
        "Survival rate of that group"
    ]
    for d in demos:
        if st.button(d, use_container_width=True):
            st.session_state.active_prompt = d

    if st.checkbox("🔍 Dataset Lab"):
        st.write("Columns:", df.columns.tolist())
        st.dataframe(df.head(5))

# ---------------- Dashboard ----------------
st.title("🚢 Titanic AI Copilot v7.5")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Sample Size", len(df))
with m2: st.metric("Survivors", df['Survived'].sum())
with m3: st.metric("Security", "Regex Sandbox")
with m4: st.metric("Reasoning", "Conversational")

st.markdown("---")

# ---------------- AI Engine ----------------
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
    """Generates high-precision Pandas code with history awareness."""
    
    history_ctx = "\n".join([f"{r}: {m}" for r, m in history[-6:]]) if history else "Start of conversation."
    
    system_prompt = f"""
    You are an expert pandas assistant for the Titanic dataset.
    Dataframe: `df`
    Columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    
    Recent History:
    {history_ctx}
    
    Rules for Semantic Precision:
    - If asked "WHO", return a row: `df.loc[df['col'].idxmax()]` or `df[df['col'] == value]`.
    - If asked "WHAT IS THE MAX/MIN", return the value: `df['age'].max()`.
    - Ratio: Count A / Count B. Use `df[df['who']=='adult'].shape[0] / df[df['who']=='child'].shape[0]`.
    - Pronouns: Resolve 'they', 'those', 'them' using the history context.
    - Return ONLY raw Python code. No markdown, no comments.
    
    Security: Never use import, os, sys, open, eval, exec.
    """
    
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        response = llm.invoke(messages).content.strip()
        if "```" in response:
            response = re.sub(r'```python|```', '', response).strip()
        return response
    except:
        return None

def is_safe(code):
    """Regex-based security with word boundaries to avoid false positives (e.g., 'Rose' containing 'os')."""
    banned_words = ["import", "open(", "exec(", "eval(", "__", "sys.", "os.", "subprocess"]
    for word in banned_words:
        # Use word boundaries \b to ensure we match the word itself, not substrings
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, code.lower()):
            return False
    return True

def safe_execute(code, df):
    """Safely execute and format the output."""
    if not code: return "The AI failed to derive a logic plan.", None
    if not is_safe(code): return f"Security Restriction: The generated logic was blocked for safety. (Reason: Banned keyword).", None

    try:
        # Use a localized context for eval
        res = eval(code, {"df": df, "pd": pd, "sns": sns, "plt": plt})
        
        fig = None
        if hasattr(res, "figure"):
            fig = res.figure
            return "Visual analysis complete:", fig
        elif isinstance(res, plt.Axes):
            fig = res.get_figure()
            return "Visual analysis complete:", fig
        
        if isinstance(res, pd.DataFrame):
            return res, None
        if isinstance(res, pd.Series):
            return res.to_frame(), None
        if isinstance(res, (float, int)):
            return f"The result is **{res:.2f}**" if isinstance(res, float) else f"The result is **{res}**", None
            
        return str(res), None
    except Exception as e:
        return f"Logic Execution Error: {e}", None

# ---------------- Chat Shell ----------------
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        if isinstance(content, pd.DataFrame): st.dataframe(content)
        elif isinstance(content, str): st.markdown(content)

user_input = st.chat_input("Ask a data question...")
if "active_prompt" in st.session_state:
    user_input = st.session_state.active_prompt
    del st.session_state.active_prompt

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        pulse = st.empty()
        pulse.markdown("""
            <div class="pulse-container">
                <div class="pulse-dot"></div><div class="pulse-dot"></div><div class="pulse-dot"></div>
                <small style="color: #94a3b8; font-family: 'Fira Code';">AI.REASONING...</small>
            </div>
        """, unsafe_allow_html=True)
        
        code = generate_reasoning_code(user_input, [m[1] for m in st.session_state.chat_history[:-1]])
        pulse.empty()
        
        # Suppress logic trace but allow for internal debugging if needed (set to False by default)
        # st.code(code, language='python') 
        
        result, fig = safe_execute(code, df)
        
        if fig:
            st.markdown(result)
            st.pyplot(fig)
            st.session_state.chat_history.append(("assistant", result))
        elif isinstance(result, pd.DataFrame):
            st.dataframe(result)
            st.session_state.chat_history.append(("assistant", "See data table above."))
        else:
            st.markdown(result)
            st.session_state.chat_history.append(("assistant", result))
