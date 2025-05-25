from dotenv import load_dotenv
import os
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
def send_feedback_to_openai(feedback: str, model: str = "gpt-3.5-turbo") -> None:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an emotion coach who gives tips based on a list of emotional states observed during a session."},
                {"role": "user", "content": feedback}
            ]
        )
        reply = response['choices'][0]['message']['content']
        return reply
    except Exception as e:
        print("Error sending feedback to OpenAI:", str(e))