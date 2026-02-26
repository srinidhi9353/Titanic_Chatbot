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
    page_title="Titanic Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium "State of the Art" look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    /* Glassmorphism containers */
    div[data-testid="stMetricValue"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="stMetricValue"]:hover {
        transform: translateY(-5px);
        border-color: #38bdf8;
    }

    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.4);
        transform: scale(1.02);
    }

    /* Chat bubble styling */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Spikes Loading Animation */
    .spikes-container {
        display: flex;
        align-items: center;
        gap: 5px;
        height: 30px;
        padding: 10px;
    }
    .spike {
        width: 4px;
        height: 10px;
        background: #38bdf8;
        border-radius: 2px;
        animation: spike-pulse 1s infinite ease-in-out;
    }
    .spike:nth-child(2) { animation-delay: 0.1s; }
    .spike:nth-child(3) { animation-delay: 0.2s; }
    .spike:nth-child(4) { animation-delay: 0.3s; }
    .spike:nth-child(5) { animation-delay: 0.4s; }

    @keyframes spike-pulse {
        0%, 100% { height: 10px; opacity: 0.5; }
        50% { height: 25px; opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
    st.title("Project Details")
    st.info("""
    **Titanic Intelligence** is an advanced data exploration tool. 
    It uses a Hybrid Analysis Engine to interact with the Titanic dataset in real-time.
    """)
    st.divider()
    st.markdown("### Example Queries")
    st.markdown("""
    - "What percentage of passengers were male?"
    - "Show me a histogram of passenger ages"
    - "What was the average ticket fare?"
    - "How many passengers embarked from each port?"
    - "Survival rate of women in 1st class"
    """)
    st.divider()
    st.markdown("### Dataset Overview")
    st.write("The dataset contains information about 891 passengers, including survival status, age, gender, and ticket class.")
    if st.checkbox("Show Raw Data"):
        st.dataframe(pd.read_csv("titanic.csv").head(10))

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    data = pd.read_csv("titanic.csv")
    return data

df = load_data()

# ---------------- Header & KPIs ----------------
st.title("🚢 Titanic Intelligence Dashboard")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Passengers", len(df))
with col2:
    survival_rate = f"{(df['Survived'].mean() * 100):.1f}%"
    st.metric("Survival Rate", survival_rate)
with col3:
    avg_age = f"{df['Age'].mean():.1f}"
    st.metric("Avg. Passenger Age", avg_age)
with col4:
    avg_fare = f"${df['Fare'].mean():.2f}"
    st.metric("Avg. Fare Paid", avg_fare)

st.markdown("---")

# ---------------- OpenRouter LLM Setup ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("Missing OpenRouter API Key. Please check your .env file.")
    st.stop()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="openai/gpt-4o-mini",
    temperature=0
)

# ---------------- Hybrid Analysis Engine ----------------

def get_intent(query):
    """Uses LLM to parse user intent into a structured plan."""
    system_prompt = """
    You are a data analyst. Parse the user's Titanic dataset query into a JSON object.
    Columns: Survived (0 or 1), Pclass (1, 2, 3), Sex (male, female), Age, SibSp, Parch, Fare, Embarked (C, Q, S).
    Return JSON format: {"intent": "stat" | "visual" | "list", "column": "col_name", "operation": "mean" | "count" | "percentage" | "distribution", "filters": {"col": "val"}}
    Examples: 
    - "male percentage" -> {"intent": "stat", "column": "Sex", "operation": "percentage", "filters": {"Sex": "male"}}
    - "age distribution" -> {"intent": "visual", "column": "Age", "operation": "distribution", "filters": {}}
    - "survival rate of women in class 1" -> {"intent": "stat", "column": "Survived", "operation": "mean", "filters": {"Sex": "female", "Pclass": 1}}
    ONLY RETURN JSON.
    """
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        response = llm.invoke(messages)
        # Extract JSON if LLM wraps it in triple backticks
        content = response.content
        if "```json" in content:
            match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if match:
                content = match.group(1)
        return json.loads(content.strip())
    except Exception as e:
        return None

def execute_analysis(plan, df):
    """Executes the analysis plan and returns (answer, fig)."""
    if not plan:
        return "I'm sorry, I couldn't parse that query. Could you try rephrasing?", None

    try:
        filtered_df = df.copy()
        for col, val in plan.get("filters", {}).items():
            if col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col] == val]

        intent = plan.get("intent")
        col = plan.get("column")
        op = plan.get("operation")

        answer = ""
        fig = None

        if intent == "stat":
            if op == "percentage":
                val = (len(filtered_df) / len(df)) * 100
                answer = f"The percentage of passengers matching those criteria is {val:.2f}%."
            elif op == "mean":
                val = filtered_df[col].mean()
                if col == "Survived":
                    answer = f"The survival rate for this group was {val*100:.2f}%."
                else:
                    answer = f"The average {col} is {val:.2f}."
            elif op == "count":
                val = len(filtered_df)
                answer = f"There were {val} passengers matching your query."

        elif intent == "visual":
            if op == "distribution" or "histogram" in str(plan):
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(data=filtered_df, x=col, kde=True, ax=ax, palette="mako")
                ax.set_title(f"Distribution of {col}")
                answer = f"Here is the distribution chart for {col}."
            elif op == "count":
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.countplot(data=filtered_df, x=col, ax=ax, palette="viridis")
                ax.set_title(f"Count of {col}")
                answer = f"Here is the frequency chart for {col}."

        if not answer:
            answer = "I found the data, but I'm not sure how to summarize it. Try asking for a 'percentage', 'average', or 'chart'."
        
        return answer, fig

    except Exception as e:
        return f"Error executing analysis: {e}", None

# ---------------- Chat Interface ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "fig" in message and message["fig"] is not None:
            st.pyplot(message["fig"])

# User Input
if prompt := st.chat_input("Ask a question about the Titanic passengers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
            <div class="spikes-container">
                <div class="spike"></div>
                <div class="spike"></div>
                <div class="spike"></div>
                <div class="spike"></div>
                <div class="spike"></div>
                <span style="margin-left: 10px; font-size: 0.9rem; color: #94a3b8;">Routing intelligence...</span>
            </div>
        """, unsafe_allow_html=True)
        
        # 1. Get Intent
        plan = get_intent(prompt)
        loading_placeholder.markdown("""
            <div class="spikes-container">
                <div class="spike"></div>
                <div class="spike"></div>
                <div class="spike"></div>
                <div class="spike"></div>
                <div class="spike"></div>
                <span style="margin-left: 10px; font-size: 0.9rem; color: #94a3b8;">Processing data...</span>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. Execute
        answer, fig = execute_analysis(plan, df)
        
        loading_placeholder.empty()
        st.markdown(answer)
        if fig:
            st.pyplot(fig)
        
        # Save to history
        new_msg = {"role": "assistant", "content": answer}
        if fig: new_msg["fig"] = fig
        st.session_state.messages.append(new_msg)
