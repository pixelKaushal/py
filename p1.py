import os
from dotenv import load_dotenv
from google import genai

import tkinter as tk

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def ask_ai():
    question = question_entry.get()
    if question.strip() == "":
        answer_label.config(text="Please enter a question.")
        return
        
    answer_label.config(text="Thinking... 🤖")
    root.update()
    
    try:
        
        
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=question + " (CRITICAL: Do not use LaTeX, markdown math formatting, or $$ symbols. Write all math and equations using plain text characters, normal numbers, and standard symbols like ^, *, /, +, -).",
            
        )
        
        answer = response.text
        answer_label.config(text=answer)
        
    except Exception as e:
        answer_label.config(text=f"Error: {e}")

root = tk.Tk()
root.title("Ai Assistant")
root.geometry("1400x800")
root.configure(bg="#17bae7")

header = tk.Label(root, text="Ask me anything!", font=("Consolas", 24, "bold"), bg="#17bae7", fg="white")
header.pack(pady=20)

question_entry = tk.Entry(root, font=("Consolas", 16), width=50)
question_entry.pack(pady=10)

ask_button = tk.Button(root, text="Ask", font=("Consolas", 16), bg="#4CAF50", fg="white", command=ask_ai)
ask_button.pack(pady=10)

answer_label = tk.Label(root, text="", font=("Consolas", 16), bg="#17bae7", fg="white", wraplength=1200, justify="left")
answer_label.pack(pady=20)

root.mainloop()