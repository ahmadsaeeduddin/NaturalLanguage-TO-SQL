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
        print(f"Table: {table}({', '.join(col_names)})")

    conn.close()
    return "\n".join(schema)

def build_prompt(nl_query, db_path):
    schema_info = extract_schema(db_path)
    prompt = f"""### Task:
Given the following database schema:

{schema_info}

Write an SQL query for:
\"{nl_query}\"
"""
    return prompt
