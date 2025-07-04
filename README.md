
# 📄 Text-to-SQL Interface Project — Full Guide

This project demonstrates how to convert natural language queries into SQL statements using a Large Language Model (LLM) from Hugging Face, and then execute those SQL queries on a sample SQLite database.

---

## 🚀 Project Goals

- ✅ Take a natural language question like **"Show names of employees older than 30"**
- ✅ Generate the appropriate SQL query automatically
- ✅ Execute it on an example SQLite database
- ✅ Return and display results

---

You have two ways to run this:
1. ✅ Using a locally **fine-tuned Hugging Face model** (`CodeT5` or similar).
2. ✅ Using the **Groq API** to call a large hosted LLM like Llama 3.

You’ll also run the generated SQL on a sample SQLite database.

---

## 📁 Recommended Project Structure

```
text_to_sql_project/
├── data/
│   └── sample.db
├── models/
│   ├── prompt_builder.py   # Builds prompt from schema
│   ├── sql_generator.py    # For local fine-tuned model
│   ├── sql_generator_groq.py # For Groq API version
│   ├── fine_tune.py        # Fine-tuning script
├── scripts/
│   └── setup_db.py         # Creates the SQLite DB
├── executor.py             # Executes SQL queries
├── .env                    # Stores API keys for Groq
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites

- Python 3.9+
- `pip install transformers datasets torch sqlparse requests python-dotenv`

---

### Install dependencies:

```bash
pip install transformers datasets torch sqlparse
```

SQLite is built-in with Python. No extra install needed.

---

## ✅ STEP 1: Create a Sample SQLite DB

In `scripts/setup_db.py`:

```python
# Creates a simple employees and departments table with dummy data.
# Run it once:

python scripts/setup_db.py
```

It will generate `data/sample.db`.

---

## ✅ STEP 2: Build the Prompt

`models/prompt_builder.py` extracts schema details:

```python
import sqlite3

def extract_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema = []
    for (table,) in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        col_names = [col[1] for col in columns]
        schema.append(f"Table: {table}({', '.join(col_names)})")

    conn.close()
    return "\n".join(schema)

def build_prompt(nl_query, db_path):
    schema_info = extract_schema(db_path)
    return f"Given schema:\n{schema_info}\nQuestion: {nl_query}"
```

---

## ✅ OPTION A — Fine-Tune Your Own Model

### 1️⃣ Prepare a dataset

Example `spider_text_sql.csv`:

`https://www.kaggle.com/datasets/mohammadnouralawad/spider-text-sql`

### 2️⃣ Fine-tune: `models/fine_tune.py`

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

df = pd.read_csv('spider_text_sql.csv')

model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

   .........
   .........
   .........

trainer.train()
trainer.save_model("./my-finetuned-codet5")
tokenizer.save_pretrained("./my-finetuned-codet5")
```

### 3️⃣ Use it in your pipeline

```python
from models.prompt_builder import build_prompt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TextToSQLGenerator:
    def __init__(self, model_name="./my-finetuned-codet5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_sql(self, prompt, max_length=128):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
```

---

## ✅ OPTION B — Use Groq API
s_function(examples):
    i
1️⃣ Store your Groq key in `.env`:

```
GROQ_API_KEY="YOUR_GROQ_KEY_HERE"
```

2️⃣ `models/sql_generator_groq.py`:

```python
import os, requests

class TextToSQLGeneratorGroq:
    def __init__(self, model_name="llama3-70b-8192"):
        self.model_name = model_name
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_sql(self, prompt):
        ..........
```

3️⃣ Example usage:

```python
from prompt_builder import build_prompt
from dotenv import load_dotenv
from models.sql_generator_groq import TextToSQLGeneratorGroq

load_dotenv()
db_path = "../data/sample.db"
nl_query = "List names of employees older than 30"
prompt = build_prompt(nl_query, db_path)

generator = TextToSQLGeneratorGroq()
print(generator.generate_sql(prompt))
```

---

## ✅ Execute the Generated SQL

**executor.py**:

```python
import sqlite3

class SQLExecutor:
    def __init__(self, db_path="data/sample.db"):
        self.db_path = db_path

    def run_query(self, sql_query):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            conn.commit()
            return {"success": True, "columns": columns, "rows": rows}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            conn.close()
```

---

## ✅ Final Best Practices

✔️ Always test with simple single-table queries first.  
✔️ Use Spider dataset for complex joins if you fine-tune.  
✔️ Use `.env` and never hardcode your keys.  
✔️ Validate generated SQL before executing!  
✔️ Consider adding a Streamlit or Flask UI later.

---

## ✅ Now you have two robust ways to build your Text-to-SQL pipeline! 🚀
