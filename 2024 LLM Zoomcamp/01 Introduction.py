from dotenv import load_dotenv
import os
from openai import OpenAI

# load the api key
load_dotenv()

# check the environment contains the key
os.environ
# if no key is input, it will take it from the environment - safer as the key
# is then never exposed in the console
client = OpenAI()
# free tier doesn't allow for API calls
response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[{"role": "user",
               "content": "is it too late to join the course?"}]
    )
response.choices[0]
response.choices[0].message
response.choices[0].message.content