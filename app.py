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
# ... (rest of the code stays same until get_intent)

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
            content = re.search(r'```json\n(.*?)\n```', content, re.DOTALL).group(1)
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
