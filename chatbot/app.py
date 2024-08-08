from flask import Flask, request, jsonify, render_template
import speech_recognition as sr
import openai
from api_key import API_key
from google.cloud import language_v1
from google.cloud import translate_v2 as translate
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_auth.json"
openai_key = API_key()
app = Flask(__name__)
recognizer = sr.Recognizer()
nlp_client = language_v1.LanguageServiceClient()
openai.api_key = openai_key.getAPIKey()

# 保存对话历史记录
conversation_history = [
    {"role": "system", "content": "you are his granddaughter."}
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/record')
def record():
    return render_template('record.html')

@app.route('/summary')
def summary():
    return render_template('summary.html')

@app.route('/start-recognition', methods=['POST'])
def start_recognition():
    global conversation_history

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            recorded_audio = recognizer.listen(source, timeout=3)
        
        text = recognizer.recognize_google(recorded_audio, language="zh-TW")
        print("Transcribed Text: {}".format(text))

        return jsonify({
            'success': True,
            'text': text
        })
    
    except sr.UnknownValueError:
        return jsonify({'success': False, 'error': 'Google Speech Recognition could not understand the audio'})
    
    except sr.RequestError as e:
        return jsonify({'success': False, 'error': f'Could not request results from Google Speech Recognition service; {e}'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process-input', methods=['POST'])
def process_input():
    global conversation_history

    user_input = request.json.get('user_input', '')
    print("User Input: {}".format(user_input))

    document = language_v1.Document(content=user_input, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = nlp_client.analyze_sentiment(document=document).document_sentiment

    # 添加用户输入至历史记录
    conversation_history.append({"role": "user", "content": user_input})

    # 调用ChatCompletion API
    completion = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0125:national-taipei-university-of-technology-csl-taipei-tech::9OUMMoCn",
        messages=conversation_history
    )
  
    chatbot_response = completion['choices'][0]['message']['content']
    zh_response = translate_text('zh-TW', chatbot_response)
    # 添加助手响应到会话历史记录
    conversation_history.append({"role": "assistant", "content": zh_response})

    return jsonify({
        'success': True,
        'text': user_input,
        'score': sentiment.score,
        'magnitude': sentiment.magnitude,
        'chatbot_response': zh_response
    })

@app.route('/summarize-conversation', methods=['POST'])
def summarize_conversation():
    global conversation_history

    # 提取用户的输入内容
    user_contents = [msg['content'] for msg in conversation_history if msg['role'] == 'user']
    user_text = "\n".join(user_contents)

    # 生成摘要
    summary = generate_summary(user_text)
    zh_summary = translate_text('zh-TW', summary)

    return jsonify({
        'success': True,
        'summary': zh_summary
    })

@app.route('/fetch-records', methods=['GET'])
def fetch_records():
    return jsonify({
        'success': True,
        'records': [{"role": msg['role'], "message": msg['content']} for msg in conversation_history if msg['role'] != 'system']
    })

def generate_summary(text):
    response = openai.Completion.create(
        # engine="text-davinci-003",
        engine = "gpt-3.5-turbo-instruct",
        prompt=(f"Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"),
        temperature=0.5,
        max_tokens=1024,  # 增加max_tokens確保摘要不被切斷
        n=1,
        stop=["\n", "Q:", "A:"]
    )
    summary = response.choices[0].text.strip()
    return summary

def translate_text(target: str, text: str) -> dict:

    translate_client = translate.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    print("Text: {}".format(result["input"]))
    print("Translation: {}".format(result["translatedText"]))
    print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result["translatedText"]


if __name__ == '__main__':
    app.run(debug=True)


