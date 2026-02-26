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
    page_title="Titanic Stable AI v8.0",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium ChatGPT-Style UI
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
    st.session_state.chat_history = []  # List of (role, content)

@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
    st.title("🚢 Stable AI v8.0")
    st.info("**Production Mode**. Fixed history unpacking, plotting crashes, and result formatting.")
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.subheader("💡 Analysis Presets")
    demos = [
        "Who is aged more?",
        "What is the maximum age?",
        "Ratio of male and female passengers",
        "Average age of 1st class survivors",
        "Is there a passenger named Rose?",
        "Survival distribution by class Chart"
    ]
    for d in demos:
        if st.button(d, use_container_width=True):
            st.session_state.active_prompt = d

    if st.checkbox("🔍 Dataset View"):
        st.dataframe(df.head(5))

# ---------------- Dashboard ----------------
st.title("🚢 Titanic Stable Copilot v8.0")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Sample Size", len(df))
with m2: st.metric("Survivors", df['Survived'].sum())
with m3: st.metric("Status", "Stable v8")
with m4: st.metric("Logic", "Semantic Express")

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
    """Generates precise Pandas code with history awareness."""
    
    # Safe history parsing to avoid ValueError Unpacking
    try:
        if history and isinstance(history[0], tuple):
            history_ctx = "\n".join([f"{r}: {m}" for r, m in history[-6:]])
        else:
            history_ctx = "Start of conversation."
    except:
        history_ctx = "Start of conversation."
    
    system_prompt = f"""
    You are an expert pandas assistant for the Titanic dataset.
    Dataframe: `df`
    Columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    
    Recent History:
    {history_ctx}
    
    Rules for Semantic Precision:
    - If asked "WHO", return a row: `df.loc[df['col'].idxmax()]` or `df[df['col'] == value]`.
    - If asked "WHAT IS THE MAX/MIN", return the value: `df['age'].max()`.
    - Ratio: Count A / Count B. Use `df[df['Sex']=='male'].shape[0] / df[df['Sex']=='female'].shape[0]`.
    - Pronouns: Resolve 'they', 'those', 'them' using the history context.
    - Generated ONLY raw Python code. No markdown, no comments.
    
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
    """Regex-based security with word boundaries."""
    banned_words = ["import", "open(", "exec(", "eval(", "__", "sys.", "os.", "subprocess"]
    for word in banned_words:
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, code.lower()):
            return False
    return True

def safe_execute(code, user_input, df):
    """Safely execute, detect plots using gcf, and format results semantic-aware."""
    if not code: return "The AI failed to derive a logic plan.", None
    if not is_safe(code): return f"Security Restriction: Logic blocked for safety.", None

    try:
        # Clear previous plots to avoid mixing
        plt.close('all')
        
        # Localized eval
        res = eval(code, {"df": df, "pd": pd, "sns": sns, "plt": plt})
        
        # Check if a plot was generated (even if res is not a plot object)
        if plt.get_fignums():
            fig = plt.gcf()
            return "Visual analysis complete:", fig
        
        # Semantic formatting for Ratios
        if "ratio" in user_input.lower() and isinstance(res, (float, int)):
            return f"The ratio is **{res:.2f} : 1**", None
            
        # Semantic formatting for Percentages
        if "percentage" in user_input.lower() and isinstance(res, (float, int)):
            return f"The result is **{res:.2f}%**", None

        # Data Outputs
        if isinstance(res, pd.DataFrame):
            return res, None
        if isinstance(res, pd.Series):
            return res.to_frame(), None
        if isinstance(res, (float, int)):
            return f"Result: **{res:.2f}**" if isinstance(res, float) else f"Result: **{res}**", None
            
        return str(res), None
    except Exception as e:
        return f"Logic Execution Error: {e}", None

# ---------------- Chat Shell ----------------
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        if isinstance(content, pd.DataFrame): st.dataframe(content)
        elif isinstance(content, str): st.markdown(content)

user_input = st.chat_input("Ask a Titanic question...")
if "active_prompt" in st.session_state:
    user_input = st.session_state.active_prompt
    del st.session_state.active_prompt

if user_input:
    # Append as tuple to prevent unpack errors in next turn
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        pulse = st.empty()
        pulse.markdown("""
            <div class="pulse-container">
                <div class="pulse-dot"></div><div class="pulse-dot"></div><div class="pulse-dot"></div>
                <small style="color: #94a3b8; font-family: 'Fira Code';">AI.STABLE_REASONING...</small>
            </div>
        """, unsafe_allow_html=True)
        
        # Pass history tuples correctly
        code = generate_reasoning_code(user_input, st.session_state.chat_history[:-1])
        pulse.empty()
        
        result, fig = safe_execute(code, user_input, df)
        
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
