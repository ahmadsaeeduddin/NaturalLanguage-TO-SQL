from models.sql_generator_usingGROQ import generate_sql_query
from executor import SQLExecutor
from dotenv import load_dotenv
import os
import re

load_dotenv()

def extract_sql(text):
    # Try to extract SQL code block
    code_block = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    # Or fallback: find SELECT ... ; statement
    match = re.search(r"(SELECT .*?;)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

db_path = "data/sample.db"
nl_query = input("Enter your question: ")

# Generate SQL
sql_query = generate_sql_query(nl_query, db_path)
print("\nGenerated SQL:\n", sql_query)

sql_query = extract_sql(sql_query)

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