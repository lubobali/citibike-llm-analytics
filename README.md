# 🚲 Citibike LLM Analytics Agent

**An AI-powered analytics agent that answers natural language questions about NYC Citibike redistribution using pre-validated SQL templates and smart LLM fallback.**

Built by **Lubo Bali** | Powered by [LuBot.ai](https://lubot.ai)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [Data Model](#data-model)
- [Features](#features)
- [Sample Questions](#sample-questions)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [AI Tools Disclosure](#ai-tools-disclosure)
- [Contact](#contact)

---

## 🎯 Overview

### The Problem
NYC Citibike needs to **redistribute bikes every morning** to meet demand. Some stations have too many bikes, others don't have enough. The question is:

> *"For July and November, how many bikes of each type should be redistributed to each station every morning during rush hour in the first week of the month?"*

### The Solution
Instead of building a one-off script, I integrated this challenge into my AI analytics platform (**LuBot.ai**) - demonstrating how I approach real production systems:

- **Ask questions in plain English** → Get instant answers
- **No SQL knowledge required** → The agent writes queries for you
- **Visual charts included** → Plotly-powered visualizations
- **Fast & reliable** → Pre-validated SQL templates (no LLM hallucination risk)

---

## 🎬 Live Demo

**Website:** [LuBot.ai](https://lubot.ai) *(deployment in progress)*

**Example Query:**
```
"For July and November, how many bikes of each type should be 
redistributed to each station every morning in the first week?"
```

**Agent Response:**
- Returns top 10 stations per month that need bikes
- Breaks down by bike type (Classic vs Electric)
- Shows comparison between July and November
- November has ~68% higher demand than July

---

## 🏗️ Architecture

![LuBot.ai Architecture](architecture.png)

### How It Works (Simple Explanation)

1. **You ask a question** → "Which stations need the most bikes in July?"

2. **Smart Router decides** → Does this match a known template? 
   - ✅ Yes → Use fast SQL template (no AI needed)
   - ❌ No → Use AI to generate custom query

3. **Database query runs** → Fetches data from Star Schema

4. **AI formats the answer** → Turns raw data into human-friendly response

5. **You see the result** → Clear answer + optional chart

### Why This Design?

| Traditional Approach | LuBot Approach |
|---------------------|----------------|
| AI generates SQL every time | Templates for common queries (faster, safer) |
| Risk of wrong queries | Pre-validated SQL (tested & proven) |
| Slow response | Sub-second for template matches |
| One-time script | Reusable platform for any data |

---

## 📊 Data Model

### Star Schema Design

I transformed the raw Citibike data into an **optimized Star Schema** - the industry standard for analytics (used by Netflix, Spotify, Amazon).

```
                    ┌─────────────────┐
                    │   fact_rides    │
                    │─────────────────│
                    │ ride_id (PK)    │
                    │ start_station   │
                    │ end_station     │
                    │ date_id (FK)    │
                    │ bike_type_id(FK)│
                    │ time_of_day_id  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐
│   dim_date   │    │dim_bike_type │    │ dim_time_of_day │
│──────────────│    │──────────────│    │─────────────────│
│ date_id (PK) │    │bike_type_id  │    │ time_of_day_id  │
│ month_name   │    │bike_type_name│    │ time_of_day_name│
│ month_num    │    │ (classic,    │    │ (Morning,       │
│ day_name     │    │  electric)   │    │  Afternoon,     │
│ day_num (1-7)│    └──────────────┘    │  Evening, Night)│
│ year         │                        └─────────────────┘
└──────────────┘
```

### Why Star Schema?

- ✅ **Fast queries** - Optimized for analytics (OLAP)
- ✅ **Easy to understand** - One fact table, simple dimensions
- ✅ **Industry standard** - Same pattern used by Fortune 500 companies
- ✅ **Flexible** - Easy to add new dimensions later

### Months Analyzed

- **July 2024** - Summer month (baseline)
- **November 2024** - Fall month (comparison)

---

## ⭐ Features

### 7 Pre-Validated SQL Templates

These templates handle the most common questions **instantly** (no AI generation needed):

| # | Template | What It Does |
|---|----------|--------------|
| 1 | `redistribution_by_station_and_type` | Bikes needed per station per type (morning rush) |
| 2 | `stations_need_bikes` | Top N stations that need more bikes |
| 3 | `stations_excess_bikes` | Top N stations with too many bikes |
| 4 | `bike_type_comparison` | Classic vs Electric usage breakdown |
| 5 | `day_of_week_shortage` | Which day has the biggest shortage |
| 6 | `busiest_stations` | Most popular stations by total rides |
| 7 | `total_rides_summary` | Overall ride counts by month |

### AGI Fallback System

If your question doesn't match a template, the AI takes over:

- **Vector Learning (FAISS)** - Finds similar past queries
- **Few-Shot RAG** - Uses successful examples to improve answers
- **User Profile Learning** - Remembers your preferences
- **Route Preference Learning** - Gets smarter over time

### Dual LLM Support

| Model | Best For |
|-------|----------|
| 🟢 **Ollama (Local)** | Privacy-sensitive data, offline use |
| ☁️ **OpenAI GPT-4o** | Best quality responses, complex queries |

---

## 💬 Sample Questions

Try asking the agent these questions:

### Redistribution Questions
```
"How many bikes should be redistributed to each station in July?"
"Which stations need the most bikes in November?"
"Compare bike redistribution between July and November"
```

### Comparison Questions
```
"Compare classic vs electric bike usage in July"
"What's the split between bike types?"
```

### Station Questions
```
"What are the top 10 busiest stations in November?"
"Which stations have excess bikes to redistribute from?"
```

### Time-Based Questions
```
"Which day of the week has the biggest shortage?"
"Show me the busiest day for bike rentals"
```

### Chart Requests
```
"Show me a bar chart of bike type usage in November"
"Visualize the top 5 stations that need bikes"
```

---

## 🛠️ Technologies Used

### Frontend
| Technology | Purpose |
|------------|---------|
| Next.js | React framework for the web interface |
| Tailwind CSS | Styling and responsive design |
| Plotly | Interactive chart visualizations |

### Backend
| Technology | Purpose |
|------------|---------|
| FastAPI | High-performance Python API framework |
| AdalFlow | LLM orchestration and agent framework |
| Python 3.11 | Core programming language |

### Database
| Technology | Purpose |
|------------|---------|
| PostgreSQL | Relational database for analytics |
| Railway | Cloud database hosting |
| Star Schema | Optimized data model for OLAP queries |

### AI/LLM
| Technology | Purpose |
|------------|---------|
| OpenAI GPT-4o | Cloud-based language model |
| Ollama (Llama) | Local/private language model |
| FAISS | Vector similarity search |

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL database
- OpenAI API key (optional - can use Ollama)

### 1. Clone the Repository
```bash
git clone https://github.com/lubobali/citibike-llm-analytics.git
cd citibike-llm-analytics
```

### 2. Set Up Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file:
```env
DATABASE_URL=postgresql://user:pass@host:port/dbname
OPENAI_API_KEY=your_key_here  # Optional if using Ollama
```

### 4. Set Up Database
```bash
# Run the schema creation script
python scripts/create_schema.py

# Load Citibike data (July & November 2024)
python scripts/load_data.py
```

### 5. Start Backend
```bash
uvicorn main:app --reload --port 8001
```

### 6. Set Up Frontend
```bash
cd frontend
npm install
npm run dev
```

### 7. Open in Browser
```
http://localhost:3000
```

---

## 📁 Project Structure

```
citibike-llm-analytics/
│
├── 📄 README.md              # This file
├── 🖼️ architecture.png       # System architecture diagram
│
├── 📂 backend/
│   ├── main.py               # FastAPI application
│   ├── agent.py              # LuBot AI agent (AdalFlow)
│   ├── requirements.txt      # Python dependencies
│   │
│   ├── 📂 tools/
│   │   ├── citibike_templates.py  # 7 SQL templates
│   │   ├── sql_tools.py           # Database utilities
│   │   └── chart_generator.py     # Plotly charts
│   │
│   └── 📂 scripts/
│       ├── create_schema.py  # Star schema creation
│       └── load_data.py      # Data loading script
│
├── 📂 frontend/
│   ├── app/
│   │   └── page.tsx          # Main chat interface
│   ├── package.json
│   └── tailwind.config.js
│
└── 📂 data/
    ├── july_2024.csv         # July Citibike data
    └── november_2024.csv     # November Citibike data
```

---

## 🤖 AI Tools Disclosure

In the spirit of transparency, here are the AI tools used in this project:

| Tool | How It Was Used |
|------|-----------------|
| **Claude (Anthropic)** | Brainstorming architecture, debugging code, README writing |
| **GitHub Copilot** | Code autocompletion and suggestions |
| **OpenAI GPT-4o** | Powers the LuBot agent's natural language responses |

The core logic, SQL templates, and system design are my own work. AI tools were used as assistants, similar to how developers use Stack Overflow or documentation.

---

## 📞 Contact

**Lubo Bali**
- 🌐 Website: [LuBot.ai](https://lubot.ai)
- 💼 Portfolio: [lubobali.com](https://lubobali.com)
- 📧 Email: [Available on request]
- 💻 GitHub: [github.com/lubobali](https://github.com/lubobali)

---

## 📜 License

This project is created for portfolio demonstration purposes.

---

*Built with ❤️ by Lubo Bali | Powered by LuBot.ai*
