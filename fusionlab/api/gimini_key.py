import os
import google.generativeai as genai

# # 1. Direct your Python traffic to your Clash Proxy
# # Replace 7890 with your Clash "Mixed Port" (check Clash settings)
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# # 2. Configure with your API Key
# GEMINI_API_KEY="AIzaSyDvXM8Mj9LbU31SVSimS2vtBKWtQsPbYnY"
# genai.configure(api_key=GEMINI_API_KEY)

# # 3. Initialize and Test
# model = genai.GenerativeModel('gemini-1.5-flash')
# try:
#     response = model.generate_content("Hello! Can you hear me through the proxy?")
#     print(f"Success: {response.text}")
# except Exception as e:
#     print(f"Error: {e}")
    
# import os
# import google.generativeai as genai

# Use the port 7890 shown in your screenshot
# 1. Direct your Python traffic to your Clash Proxy
# Replace 7890 with your Clash "Mixed Port" (check Clash settings)

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

GEMINI_API_KEY="AIzaSyDvXM8Mj9LbU31SVSimS2vtBKWtQsPbYnY"
genai.configure(api_key=GEMINI_API_KEY)

# Explicitly use the stable version
model = genai.GenerativeModel('gemini-1.5-flash-latest') 

try:
    response = model.generate_content("Testing connection from Changsha.")
    print(f"Success: {response.text}")
except Exception as e:
    print(f"Error: {e}")