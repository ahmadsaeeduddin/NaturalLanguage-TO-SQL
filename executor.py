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
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            conn.commit()
            conn.close()

            return {
                "success": True,
                "columns": columns,
                "rows": rows
            }
        except Exception as e:
            conn.close()
            return {
                "success": False,
                "error": str(e)
            }


# if __name__ == "__main__":
#     executor = SQLExecutor()
    
#     test_query = "SELECT name, age FROM employees WHERE age > 30;"
#     result = executor.run_query(test_query)
    
#     if result["success"]:
#         print("✅ Query executed successfully.")
#         print("Columns:", result["columns"])
#         for row in result["rows"]:
#             print(row)
#     else:
#         print("❌ Query failed:", result["error"])
