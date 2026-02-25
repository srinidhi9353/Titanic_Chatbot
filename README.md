# 🚢 Titanic Intelligence: Natural Language Data Assistant

A sophisticated data analysis tool that allows users to interact with the classic Titanic dataset using plain English. Built with **LangChain**, **Streamlit**, and **OpenRouter**, this project demonstrates the power of LLM-driven data exploration and automated visualization.

## 🚀 Overview

Traditional data analysis often requires writing complex SQL or Python code. This project bridges that gap by using a custom LangChain agent that interprets natural language queries, executes appropriate data manipulations in Pandas, and generates relevant insights and visualizations on the fly.

## 🛠️ Tech Stack

- **Streamlit**: For building the interactive data dashboard.
- **LangChain**: For orchestrating the data agent and LLM logic.
- **OpenRouter**: Leveraging the **GPT-4o mini** model for high-performance reasoning and data synthesis.
- **Pandas/Seaborn**: For advanced data manipulation and statistical visualization.

## ✨ Key Features

- **Natural Language Querying**: Ask questions like "What was the survival rate of first-class passengers?" or "Analyze the age distribution."
- **Automated Data Processing**: The agent handles missing values and data types automatically to provide accurate answers.
- **Dynamic Visualizations**: Automatically generates histograms and bar charts based on the context of the user's question.
- **Cached Data Loading**: Optimized performance using Streamlit's caching mechanism.

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.8+
- An OpenRouter API Key

### 2. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory and add your OpenRouter API key:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Running the Application
Launch the Streamlit server:
```bash
streamlit run app.py
```

## 📈 Example Queries

- "What is the overall survival rate?"
- "Compare the average fare paid by survivors vs non-survivors."
- "Show me a histogram of passenger ages."
- "How many people embarked from each city?"

---
*Developed as a showcase of LLM-integrated data science tools.*
