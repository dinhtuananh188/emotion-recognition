import requests
import logging
from pydub import AudioSegment

def convert_to_wav_mono_16k(src_path, dst_path):
    """
    Converts any audio file to WAV format with mono channel and 16kHz sample rate.
    """
    sound = AudioSegment.from_file(src_path)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    sound.export(dst_path, format="wav")


def transcribe_audio(audio_path, token):
    """
    Sends an audio file to Viettel's STT API for transcription.

    Args:
        audio_path (str): Path to the audio file.
        token (str): Shared token for authentication.

    Returns:
        str: Transcribed text from the audio file.
    """
    url = "https://viettelai.vn/asr/recognize"
    payload = {'token': token}
    headers = {
        'accept': '*/*'
    }

    try:
        with open(audio_path, 'rb') as f:
            files = [
                ('file', (audio_path, f, 'audio/wav'))
            ]
            response = requests.request("POST", url, headers=headers, data=payload, files=files)

            response.raise_for_status()

            # Parse the API response
            api_response = response.json()

            # Check if the 'code' is 200, then return the 'transcript'
            if api_response.get('code') == 200:
                transcript = api_response.get('response', {}).get('result', [{}])[0].get('transcript', '')
                logging.info(f"Transcription: {transcript}")
                return transcript
            else:
                logging.error(f"API error: {api_response.get('message')}")
                return ""
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during STT API call: {e}")
        return ""
