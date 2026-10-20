import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from app.chat.chatbot import process_chat_message

prompt = "backtest the all the pairs last 15 days M3 ,1:3"
if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])

print(f"\n[+] Running Chatbot Query: '{prompt}' ...\n")
res = process_chat_message(prompt)
print(res.get("reply", ""))
