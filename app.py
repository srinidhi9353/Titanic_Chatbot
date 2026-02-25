
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
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
    It uses a LangChain-powered agent to interact with the Titanic dataset in real-time.
    """)
    st.divider()
    st.markdown("### Example Queries")
    st.markdown("""
    - "What percentage of passengers were male?"
    - "Show me a histogram of passenger ages"
    - "What was the average ticket fare?"
    - "How many passengers embarked from each port?"
    """)
    st.divider()
    st.markdown("### Dataset Overview")
    st.write("The dataset contains information about 891 passengers, including survival status, age, gender, and ticket class.")
    if st.checkbox("Show Raw Data"):
        st.dataframe(pd.read_csv("titanic.csv").head(10))

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    # Loading local CSV file
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
    # Use 'Survived' (capitalized) from CSV
    survival_rate = f"{(df['Survived'].mean() * 100):.1f}%"
    st.metric("Survival Rate", survival_rate)
with col3:
    # Use 'Age' (capitalized) from CSV
    avg_age = f"{df['Age'].mean():.1f}"
    st.metric("Avg. Passenger Age", avg_age)
with col4:
    # Use 'Fare' (capitalized) from CSV
    avg_fare = f"${df['Fare'].mean():.2f}"
    st.metric("Avg. Fare Paid", avg_fare)

st.markdown("---")

# ---------------- LLM Setup ----------------
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

# Enhanced Agent Brain
prefix = """
You are an expert data science assistant specializing in the Titanic dataset.
Your goal is to provide accurate, concise, and professional answers to passenger-related queries.
When answering:
1. Provide a clear text summary of the findings.
2. If the user asks for a chart, plot, or distribution, tell them what you found and then let the UI handle the visualization.
3. Be robust to different question formats. If a query is ambiguous, explain your assumptions.
4. Always refer to the data columns accurately (e.g., 'Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked').
"""

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    prefix=prefix,
    allow_dangerous_code=True
)

# ---------------- Chat Interface ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
                <span style="margin-left: 10px; font-size: 0.9rem; color: #94a3b8;">Analyzing patterns...</span>
            </div>
        """, unsafe_allow_html=True)
        
        try:
            response = agent.run(prompt)
            loading_placeholder.empty()
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Context-aware visualization
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ["plot", "chart", "visualize", "distribution", "histogram", "graph"]):
                st.divider()
                st.caption("Auto-generated visualization based on your query:")
                
                if "age" in prompt_lower:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.histplot(data=df, x="Age", hue="Survived", kde=True, palette="mako", ax=ax)
                    st.pyplot(fig)
                
                elif "class" in prompt_lower or "pclass" in prompt_lower:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.countplot(data=df, x="Pclass", hue="Survived", palette="viridis", ax=ax)
                    st.pyplot(fig)
                
                elif "gender" in prompt_lower or "sex" in prompt_lower or "male" in prompt_lower or "female" in prompt_lower:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=df, x="Sex", y="Survived", palette="magma", ax=ax)
                    st.pyplot(fig)
                
                elif "port" in prompt_lower or "embark" in prompt_lower:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.countplot(data=df, x="Embarked", palette="cubehelix", ax=ax)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error analyzing data: {e}")
