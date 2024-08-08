import requests

BASE_URL = "http://0.0.0.0:8010" 
translate_url = f"{BASE_URL}/translate/"
generate_url = f"{BASE_URL}/generate/"
analyse_url =f"{BASE_URL}/sentiment_analysis/"
chat_url = f"{BASE_URL}/chat_model/"
sql_url = f"{BASE_URL}/generate-sql/"

def test_index():
    response = requests.get(f"{BASE_URL}/")
    print(response.json())

def test_transcribe_audio():
    with open("english.mp3", "rb") as audio_file:
        files = {"Audio_file": ("audiofile.mp3", audio_file, "audio/mp3")}
        response = requests.post(f"{BASE_URL}/transcribe/", files=files)
        print(response.json())

def test_text_translate():
    data = {
    "text": "Hello, how are you?" ,
    "target_lang": "fr",
    }
    response = requests.post(translate_url, json=data)
    translated_text = response.json()
    print(f"Translated Text: {translated_text}")

def test_generate_text():
    data = {"prompt" : "Explanation : Who is Narendra Modi"}
    response = requests.post(generate_url, json=data)
    print(response.json())

def test_sentiment_analysis():
    data= {"prompt" : "I love this product!"}
    response = requests.post(analyse_url, json=data)
    print(response.json())

def test_chat_with_model():
    data = {"prompt" : "Who is the president of USA?"}
    response = requests.post(chat_url, json=data)
    print(response.json())
    
def test_sql_model():
    data =  {"prompt": "show all the products with the price greater than 100"}
    response = requests.post(sql_url, json=data)
    print(response.json())
    
def get_posted_text():
    response = requests.get(f"{BASE_URL}/get_the_data/")
    print(response.json())

if __name__ == "__main__":
    test_index()
    test_transcribe_audio()
    test_text_translate()
    test_generate_text()
    test_sentiment_analysis()
    test_chat_with_model()
    test_sql_model()
    get_posted_text()