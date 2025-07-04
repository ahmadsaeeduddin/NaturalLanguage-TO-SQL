
# 📄 Text-to-SQL Interface using Hugging Face LLMs

This project demonstrates how to convert natural language queries into SQL statements using a Large Language Model (LLM) from Hugging Face, and then execute those SQL queries on a sample SQLite database.

---

## 🚀 Project Goals

- ✅ Take a natural language question like **"Show names of employees older than 30"**
- ✅ Generate the appropriate SQL query automatically
- ✅ Execute it on an example SQLite database
- ✅ Return and display results

---

## 📁 Project Structure

```
text_to_sql_project/
├── data/
│   └── sample.db          # SQLite database file
├── models/
│   ├── prompt_builder.py  # Builds prompt for LLM
│   └── sql_generator.py   # Loads LLM and generates SQL
├── scripts/
│   └── setup_db.py        # Creates and populates sample database
├── executor.py            # Executes generated SQL
├── app.py                 # (Optional) Main CLI or UI script
├── requirements.txt       # Python dependencies
└── .env                   # (Optional) for API keys
```

---

## ⚙️ Prerequisites

- **Python 3.9+**
- Basic understanding of NLP and SQL

### Install dependencies:

```bash
pip install transformers datasets torch sqlparse
```

SQLite is built-in with Python. No extra install needed.

---

## ✅ STEP 1: Set Up Sample Database

Create and populate a simple SQLite database with tables `employees` and `departments`.

**File:** `scripts/setup_db.py`

Run:

```bash
python scripts/setup_db.py
```

This generates `data/sample.db` with some dummy data.

---

## ✅ STEP 2: Build Prompt Template

**File:** `models/prompt_builder.py`

This script:

1. Reads the SQLite schema (`PRAGMA table_info`).
2. Builds a structured prompt for the LLM.

Example output:

```
### Task:
Given the following database schema:

Table: employees(id, name, age, department_id)
Table: departments(id, name)

Write an SQL query for:
"Show the names of employees older than 30"
```

Test it by importing `build_prompt` in Python.

---

## ✅ STEP 3: Load LLM & Generate SQL

**File:** `models/sql_generator.py`

This script:

1. Loads a Hugging Face model (e.g., `Salesforce/codet5-small`).
2. Encodes the prompt.
3. Generates SQL.

**Important:** `t5-small` is generic — use `codet5` or a fine-tuned Text-to-SQL model for better results.

Example usage:

```python
from models.prompt_builder import build_prompt
from models.sql_generator import TextToSQLGenerator

db_path = "data/sample.db"
nl_query = "List names of employees older than 30"

prompt = build_prompt(nl_query, db_path)
prompt = f"translate English to SQL: {prompt}"

generator = TextToSQLGenerator(model_name="Salesforce/codet5-small")
sql_query = generator.generate_sql(prompt)

print(sql_query)
```

---

## ✅ STEP 4: Execute Generated SQL

**File:** `executor.py`

This script:

1. Connects to `sample.db`
2. Runs the generated SQL
3. Returns rows or an error

Example usage:

```python
from executor import SQLExecutor

executor = SQLExecutor()
result = executor.run_query("SELECT name, age FROM employees WHERE age > 30;")

if result["success"]:
    print("Columns:", result["columns"])
    for row in result["rows"]:
        print(row)
else:
    print("Error:", result["error"])
```

---

## ✅ STEP 5: End-to-End (Optional)

**You can combine steps into `app.py`:**

```python
from models.prompt_builder import build_prompt
from models.sql_generator import TextToSQLGenerator
from executor import SQLExecutor

db_path = "data/sample.db"
nl_query = input("Enter your question: ")

# Build prompt
prompt = build_prompt(nl_query, db_path)
prompt = f"translate English to SQL: {prompt}"
print("\nPrompt:\n", prompt)

# Generate SQL
generator = TextToSQLGenerator(model_name="Salesforce/codet5-small")
sql_query = generator.generate_sql(prompt)
print("\nGenerated SQL:\n", sql_query)

# Execute SQL
executor = SQLExecutor()
result = executor.run_query(sql_query)

print("\nResult:")
if result["success"]:
    print("Columns:", result["columns"])
    for row in result["rows"]:
        print(row)
else:
    print("Error:", result["error"])
```

Run:

```bash
python app.py
```

✅ This will prompt you for a question, generate SQL, run it, and show results.

---

## 🧩 Tips & Next Steps

- 🔒 Use a `.env` file if you use private API keys (e.g., OpenAI API).
- 🗃️ Use DB Browser for SQLite to inspect your database.
- 🔬 Fine-tune your model on Text-to-SQL data for better accuracy.
- 🖥️ Add a Streamlit or Flask frontend if you want a web UI.

---

## ✅ Useful Links

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Salesforce CodeT5](https://huggingface.co/Salesforce/codet5-small)
- [SQLite Browser](https://sqlitebrowser.org/)

---

## 🏁 That’s It!

You now have a working Text-to-SQL pipeline:
- Natural Language → Prompt → SQL → Execute.

Happy coding!
