from models.prompt_builder import build_prompt

db_path = "data/sample.db"
nl_query = "Show the names of employees older than 30"
prompt = build_prompt(nl_query, db_path)

print(prompt)
