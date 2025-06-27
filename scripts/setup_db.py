import sqlite3

# Create connection
conn = sqlite3.connect("data/sample.db")
cursor = conn.cursor()

# Drop tables if they exist
cursor.execute("DROP TABLE IF EXISTS employees;")
cursor.execute("DROP TABLE IF EXISTS departments;")

# Create tables
cursor.execute("""
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
""")

cursor.execute("""
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    department_id INTEGER,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);
""")

# Insert data
departments = [
    (1, "Engineering"),
    (2, "Sales"),
    (3, "HR"),
    (4, "Marketing")
]
employees = [
    (1, "Alice", 30, 1),
    (2, "Bob", 25, 2),
    (3, "Charlie", 40, 1),
    (4, "Diana", 35, 3),
    (5, "Ethan", 28, 4),
    (6, "Fiona", 45, 2)
]

cursor.executemany("INSERT INTO departments VALUES (?, ?);", departments)
cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?);", employees)

conn.commit()
conn.close()

print("✅ SQLite database with employees and departments created successfully.")
