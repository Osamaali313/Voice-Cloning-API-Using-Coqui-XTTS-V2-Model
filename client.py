import argparse
import requests
import base64
from datetime import datetime

# Update this URL to your server's URL if hosted remotely
API_URL = "http://127.0.0.1:8000/predict"

def send_text_to_speech_request(text):
    response = requests.post(API_URL, json={"text": text})
    if response.status_code == 200:
        audio_content_base64 = response.json()["audio_content"]
        audio_content = base64.b64decode(audio_content_base64)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").lower()
        filename = f"tts_output_{timestamp}.wav"
        
        with open(filename, "wb") as audio_file:
            audio_file.write(audio_content)
        
        print(f"Audio saved to {filename}")
    else:
        print(f"Error: Response with status code {response.status_code} - {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send text to TTS server and receive synthesized speech.")
    parser.add_argument("--text", required=True, help="Text to synthesize into speech.")
    args = parser.parse_args()
    
    send_text_to_speech_request(args.text)