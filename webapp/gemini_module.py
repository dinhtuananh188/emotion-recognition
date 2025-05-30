import google.generativeai as genai
from uuid import uuid4
from flask import session
import requests
import re
import os
import json

GENAI_API_KEY = os.getenv('GENAI_API_KEY')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Thông tin cấu hình ViettelAI TTS
VIETTEL_TTS_API_KEY = os.getenv('VIETTEL_TTS_API_KEY')

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def youtube_search(query, max_results=1):
    try:
        search_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "key": YOUTUBE_API_KEY,
            "maxResults": max_results,
            "type": "video",
            "regionCode": "VN",
            "relevanceLanguage": "vi"
        }
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        results = response.json()
        items = results.get("items", [])
        if items:
            video = items[0]

            video_id = video["id"]["videoId"]
            snippet = video["snippet"]
            return {
                "videoId": video_id,
                "title": snippet["title"],
                "description": snippet["description"],
                "channelTitle": snippet["channelTitle"],
                "publishedAt": snippet["publishedAt"],
                "url": f"https://www.youtube.com/watch?v={video_id}"
            }
        return None
    except Exception as e:
        print(f"[YouTube Error] {e}")
        return None

def generate_tts_audio(text, index):
    try:
        url = "https://viettelai.vn/tts/speech_synthesis"
        payload = json.dumps({
            "text": text,
            "voice": "hcm-diemmy",
            "speed": 1,
            "tts_return_option": 3,
            "token": VIETTEL_TTS_API_KEY,
            "without_filter": False
        })
        headers = {
            'accept': '*/*',
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            tts_folder = "webapp/static/audio"
            os.makedirs(tts_folder, exist_ok=True)
            filename = f"tts_{index}.mp3"
            filepath = os.path.join(tts_folder, filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            return f"/{tts_folder}/{filename}"
        else:
            print(f"[TTS Error] Status: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        print(f"[TTS Exception] {e}")
        return None

def classify_intent(user_input):
    try:
        prompt = (
            f"Bạn là một trợ lý AI. Phân tích yêu cầu sau và trả lời duy nhất một từ:\n"
            f"Nếu yêu cầu mang tính cần nghe nhạc, trả lời 'giải trí'.\n"
            f"Còn không phải, trả lời 'khác'\n\n"
            f"Yêu cầu: \"{user_input}\"\nKết quả:"
        )
        result = model.generate_content(prompt)
        intent = result.text.strip().lower()
        return intent
    except Exception as e:
        print(f"[Intent Error] {e}")
        return "khác"

def handle_request(user_input, session):
    # Lấy danh sách video và lịch sử chat trong session
    chat_history = session.get('chat_history', [])
    sent_videos = session.get('sent_videos', [])

    if 'chat_history' not in session:
    # Clear old TTS files if any
        tts_folder = "webapp/static/audio"
        if os.path.exists(tts_folder):
            for file in os.listdir(tts_folder):
                if file.endswith(".mp3"):
                    os.remove(os.path.join(tts_folder, file))
        session['tts_files'] = []


    # Xác định ý định người dùng
    intent = classify_intent(user_input)

    if "giải trí" in intent:
        if "đổi bài" in user_input.lower() and sent_videos:
            # Nếu yêu cầu là đổi bài, thì tìm video mới chưa gửi
            query = chat_history[-1] if chat_history else "video giải trí"
        else:
            query = user_input

        # Gọi API YouTube
        search_results = []
        for i in range(5):  # lấy tối đa 5 video để chọn video chưa gửi
            result = youtube_search(query, max_results=i+1)
            if result and result['videoId'] not in sent_videos:
                search_results.append(result)
                break

        if search_results:
            selected_video = search_results[0]
            sent_videos.append(selected_video['videoId'])

            # Lưu lại
            session['sent_videos'] = sent_videos
            if "đổi bài" not in user_input.lower():
                chat_history.append(user_input)
                session['chat_history'] = chat_history

            return {
                "type": "youtube",
                "response": f"🎵 {selected_video['title']}\n📺 {selected_video['url']}"
            }
        else:
            return {
                "type": "error",
                "response": "Không tìm thấy video mới chưa từng gửi."
            }
    else:
        try:
            # Tạo phiên chat mới với Gemini
            chat = model.start_chat(history=[
                {"role": "user", "parts": [msg]} for msg in chat_history
            ])
            prompt = (
                "Bạn là một trợ lý AI. Hãy trả lời câu hỏi sau bằng tiếng Việt, "
                "trừ khi câu hỏi được đặt bằng tiếng Anh, và giới hạn tối đa 300 ký tự:\n"
                f"{user_input}"
            )
            response = chat.send_message(prompt)
            raw_text = response.text.strip()

            # Cập nhật lịch sử
            chat_history.append(user_input)
            session['chat_history'] = chat_history

            def format_response(raw_text):
                # Split paragraphs more distinctly
                formatted_response = re.sub(r'(?<=[.!?])\s+(?=[A-ZÀ-Ỵ])', '\n\n', raw_text)

                # Replace asterisks (*) with bullet points or line breaks
                formatted_response = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', formatted_response)  # Bold for double asterisks
                formatted_response = re.sub(r'\*([^*]+)\*', r'- \1', formatted_response)  # Bullet points for single asterisks

                # Add extra formatting for better readability
                formatted_response = f"🤖 {formatted_response.strip()}"

                # Ensure consistent line breaks for better display
                formatted_response = formatted_response.replace('\n', '<br>')

                return formatted_response

            formatted_response = format_response(raw_text)
            # Generate TTS
            tts_index = len(session.get('tts_files', []))
            tts_url = generate_tts_audio(raw_text, tts_index)
            if tts_url:
                tts_files = session.get('tts_files', [])
                tts_files.append(tts_url)
                session['tts_files'] = tts_files

            if tts_url:
                tts_url = tts_url.replace("webapp/", "")
            print(tts_url)

            return {
                "type": "gemini",
                "response": formatted_response,
                "tts_url": tts_url
            }

        except Exception as e:
            print(f"[Gemini Error] {e}")
            return {
                "type": "error",
                "response": "Đã xảy ra lỗi khi sinh nội dung từ AI."
            }
