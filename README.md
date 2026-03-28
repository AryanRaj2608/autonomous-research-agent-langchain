# 🔬 Assignment 2: Autonomous Research Agent

An autonomous research agent built with **LangChain** that searches the web, gathers knowledge from multiple sources, analyzes data, and generates a structured research report on any user-specified topic.

## 🚀 Features

- **Web Search** — Uses Tavily Search API for real-time web information
- **Wikipedia Integration** — Queries Wikipedia for encyclopedic knowledge
- **ReAct Reasoning** — Uses Reasoning + Acting strategy for intelligent research
- **Structured Reports** — Generates well-organized research reports
- **Auto-Save** — Saves reports to the `outputs/` folder automatically

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | Agent framework and orchestration |
| ChatGroq (llama-3.3-70b-versatile) | Large Language Model |
| Tavily Search | Web search tool |
| Wikipedia API | Encyclopedic knowledge tool |
| ReAct Agent | Reasoning + Acting strategy |

## 📁 Project Structure

```
ai assignment 2/
│
├── autonomous_research_agent.py   # Main agent script
├── tools.py                       # Search tools (Tavily + Wikipedia)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
└── outputs/                       # Generated research reports
    └── sample_outputs.txt         # Sample output placeholder
```

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/autonomous-research-agent-langchain.git
cd autonomous-research-agent-langchain
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

- Get your Groq API key from: [https://console.groq.com/](https://console.groq.com/)
- Get your Tavily API key from: [https://tavily.com/](https://tavily.com/)

## ▶️ Usage

Run the research agent:

```bash
python autonomous_research_agent.py
```

The agent will:

1. Ask you for a research topic
2. Search the web using Tavily for current information
3. Query Wikipedia for background knowledge
4. Analyze and cross-reference gathered data
5. Generate a structured research report
6. Display the report in the terminal
7. Save the report to the `outputs/` folder

## 📄 Output Format

The generated report follows this structure:

```
# Research Report: [Topic]

## 1. Introduction
## 2. Key Insights
## 3. Applications
## 4. Advantages
## 5. Challenges
## 6. Conclusion
```

## 🏗️ Architecture

```
User Input (Topic)
       │
       ▼
┌──────────────────┐
│   ReAct Agent    │
│  (AgentExecutor) │
│                  │
│  Thought → Act   │
│  → Observe →     │
│  Repeat          │
└──────┬───────────┘
       │
       ├──── Tavily Search (Web)
       │
       ├──── Wikipedia (Knowledge)
       │
       ▼
┌──────────────────┐
│  ChatGroq LLM    │
│ (llama3-70b-8192)│
└──────┬───────────┘
       │
       ▼
  Structured Report
  (Display + Save)
```

## 📋 Requirements

- Python 3.9+
- Groq API Key
- Tavily API Key
- Internet connection

## 👤 Author

**Aditya Raj**

## 📝 License

This project is created for academic purposes as part of Assignment 2.
