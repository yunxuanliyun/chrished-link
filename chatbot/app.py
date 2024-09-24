import requests
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import speech_recognition as sr
import openai
from api_key import API_key
from google.cloud import language_v1
from google.cloud import translate_v2 as translate
import os
import json
import sqlite3
from datetime import datetime

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_auth.json"
# Lab server
url = "http://140.124.94.72:6000/chat/"
app = Flask(__name__)
recognizer = sr.Recognizer()
nlp_client = language_v1.LanguageServiceClient()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 初始化db
def init_db():
    conn = sqlite3.connect('conversation.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# 保存歷史紀錄到db
def save_message_to_db(sender, content):
    conn = sqlite3.connect('conversation.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversation (sender, content) VALUES (?, ?)
    ''', (sender, content))
    conn.commit()
    conn.close()

# 從db獲取對話紀錄
def get_conversation_history(date=None):
    conn = sqlite3.connect('conversation.db')
    cursor = conn.cursor()
    
    if date:
        # 獲取指定日期的紀錄(當天)
        cursor.execute('''
            SELECT sender, content, timestamp FROM conversation
            WHERE DATE(timestamp) = ?
            ORDER BY id
        ''', (date,))
    else:
        # 獲取所有紀錄
        cursor.execute('''
            SELECT sender, content, timestamp FROM conversation ORDER BY id
        ''')
    
    rows = cursor.fetchall()
    conversation_history = [{'sender': row[0], 'content': row[1], 'timestamp': row[2]} for row in rows]
    conn.close()
    return conversation_history

# 初始化db
init_db()

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
    records = get_conversation_history()
    return render_template('record.html', records=records)

@app.route('/summary')
def summary():
    return render_template('summary.html')

@app.route('/start-recognition', methods=['POST'])
def start_recognition():
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
    user_input = request.json.get('user_input', '')
    print("User Input: {}".format(user_input))

    # 保存用户input至db
    save_message_to_db("user", user_input)

    # 情感分析
    document = language_v1.Document(content=user_input, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = nlp_client.analyze_sentiment(document=document).document_sentiment

    url = "http://140.124.94.72:6000/chat/"

    data = {
        "input_text": user_input  # 用戶輸入的文本
        # 不再包含 "context"
    }

    # 發送HTTP请求到lab server
    response = requests.post(url, json=data)

    # 解析API的回應
    if response.status_code == 200:
        result = response.json()
        chatbot_response = result['answer']

        # 將回應翻譯成中文，僅保留翻譯成果
        zh_response = translate_text('zh-TW', chatbot_response)

    else:
        zh_response = "抱歉，我無法取得回應。"

    # 保存chatbot回覆至db
    save_message_to_db("bot", zh_response)

    return jsonify({
        'success': True,
        'text': user_input,
        'score': round(sentiment.score, 2),
        'magnitude': round(sentiment.magnitude, 2),
        'chatbot_response': zh_response
    })

@app.route('/summarize-conversation', methods=['POST'])
def summarize_conversation():
    # 獲得今天的日期，格式為'YYYY-MM-DD'
    today = datetime.now().strftime('%Y-%m-%d')

    # 獲得今天的對話紀錄
    conversation_history = get_conversation_history(date=today)

    # 檢查今天是否有對話紀錄
    if not conversation_history:
        return jsonify({
            'success': False,
            'summary': '今天没有對話紀錄。'
        })

    # 將對話內容轉換為文本形式，格式為 "使用者: xxx\n機器人: xxx"
    conversation_text = ""
    for msg in conversation_history:
        if 'content' in msg and 'sender' in msg:
            if msg['sender'] == 'user':
                conversation_text += f"使用者: {msg['content']}\n"
            elif msg['sender'] == 'bot':
                conversation_text += f"機器人: {msg['content']}\n"
            else:
                conversation_text += f"{msg['content']}\n"

    # 将對話內容翻譯成英文
    en_conversation_text = translate_text('en', conversation_text)

    # 將對話內容進行切割，避免超過模型的輸入限制
    max_chunk_size = 1000  # 根據模型的限制調整
    text_chunks = []
    while len(en_conversation_text) > 0:
        chunk = en_conversation_text[:max_chunk_size]
        en_conversation_text = en_conversation_text[max_chunk_size:]
        text_chunks.append(chunk)

    # 對每塊進行摘要
    summaries = []
    for chunk in text_chunks:
        summary = generate_summary(chunk)
        if summary:
            summaries.append(summary)

    # 合併摘要並翻譯回中文
    full_summary_en = " ".join(summaries)
    zh_summary = translate_text('zh-TW', full_summary_en)

    return jsonify({
        'success': True,
        'summary': zh_summary
    })

@app.route('/fetch-records', methods=['GET'])
def fetch_records():
    # 獲取所有對話紀錄
    conversation_history = get_conversation_history()
    return jsonify({
        'success': True,
        'records': [{"sender": msg.get('sender', ''), "message": msg.get('content', ''), "timestamp": msg.get('timestamp', '')} for msg in conversation_history]
    })

def generate_summary(text):
    try:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        # 检查摘要是否為默認回覆或無效內容
        default_replies = [
            "I'm sorry, but I cannot provide a summary.",
            "Sorry, I cannot summarize this text.",
            "抱歉，我無法總結。",
            # 可以根據實際情况添加更多默認回覆
        ]
        if summary.strip() == "" or summary.strip() in default_replies:
            return None
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

def translate_text(target: str, text: str) -> str:
    translate_client = translate.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # 翻譯文本
    result = translate_client.translate(text, target_language=target)

    # 只返回翻譯後的文本，過濾掉其他內容
    translated_text = result["translatedText"]

    return translated_text

if __name__ == '__main__':
    app.run(debug=True)