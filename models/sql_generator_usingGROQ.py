from prompt_builder import build_prompt
from sql_generator import TextToSQLGeneratorGroq
import os

def generate_sql_query(nl_query, db_path = "../data/sample.db"):
    schema_prompt = build_prompt(nl_query, db_path)
    prompt = f"""Given the following database schema:\n{schema_prompt}\nWrite an SQL query for the above question."""
    generator = TextToSQLGeneratorGroq(model_name="llama3-70b-8192")
    sql_query = generator.generate_sql(prompt)
    return sql_query
