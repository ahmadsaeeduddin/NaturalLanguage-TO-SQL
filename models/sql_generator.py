# from prompt_builder import build_prompt
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# class TextToSQLGenerator:
#     def __init__(self, model_name="t5-small"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     def generate_sql(self, prompt, max_length=128):
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         output = self.model.generate(
#             **inputs,
#             max_length=max_length,
#             num_beams=4,
#             early_stopping=True
#         )
#         sql_query = self.tokenizer.decode(output[0], skip_special_tokens=True)
#         return sql_query.strip()

# if __name__ == "__main__":

#     db_path = "../data/sample.db"
#     nl_query = "List names of employees older than 30"
#     bu_prompt = build_prompt(nl_query, db_path)

#     generator = TextToSQLGenerator(model_name="Salesforce/codet5-base")
#     sql_query = generator.generate_sql(bu_prompt)

#     print("Generated SQL:")
#     print(sql_query)

import os
import requests

class TextToSQLGeneratorGroq:
    def __init__(self, model_name="llama3-70b-8192"):
        self.model_name = model_name
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_sql(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that converts English questions into syntactically correct SQL queries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": messages
        }

        response = requests.post(self.base_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            sql_query = result["choices"][0]["message"]["content"].strip()
            return sql_query
        else:
            print("❌ Error:", response.status_code, response.text)
            return None

