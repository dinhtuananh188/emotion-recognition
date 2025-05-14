// Hàm xử lý khi có lỗi camera
function handleCameraError() {
    const cameraContainer = document.querySelector('.camera-container');
    cameraContainer.innerHTML = '<p class="error">Camera không khả dụng</p>';
}

// Thêm style cho thông báo lỗi
const style = document.createElement('style');
style.textContent = `
    .error {
        color: red;
        text-align: center;
        padding: 20px;
    }
    .message.ai iframe {
        max-width: 100%;
        width: 560px;
        height: 315px;
        display: block;
        margin: 10px auto;
        border-radius: 12px;
    }
    .video-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 10px 0;
    }
    .video-title {
        margin-bottom: 10px;
        font-weight: bold;
        max-width: 560px;
        text-align: center;
    }
`;
document.head.appendChild(style);

// Hàm trích xuất ID video YouTube và tiêu đề
function extractYouTubeInfo(url) {
    const youtubeRegex = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
    const match = url.match(youtubeRegex);
    
    if (match) {
        const cleanedUrl = url.replace(match[0], '').trim();
        return {
            videoId: match[1],
            title: cleanedUrl || 'YouTube Video'
        };
    }
    return null;
}

// Cập nhật hàm phát âm thanh với nút nằm ngoài message
function handleAudioPlayback(ttsUrl, aiMessage) {
    const audio = new Audio(ttsUrl);

    const wrapper = document.createElement('div');
    wrapper.className = 'message-wrapper';

    const playPauseButton = document.createElement('button');
    playPauseButton.innerHTML = '⏸';
    playPauseButton.className = 'audio-button';

    let isPlaying = true;
    audio.play();

    playPauseButton.addEventListener('click', () => {
        if (isPlaying) {
            audio.pause();
            playPauseButton.innerHTML = '▶';
        } else {
            audio.play();
            playPauseButton.innerHTML = '⏸';
        }
        isPlaying = !isPlaying;
    });

    wrapper.appendChild(playPauseButton);
    wrapper.appendChild(aiMessage); // aiMessage là div.message.ai

    const chatMessages = document.getElementById('chat-messages');
    chatMessages.appendChild(wrapper);
}


document.addEventListener('DOMContentLoaded', async () => {
    
    try {
        const response = await fetch('/api/track_yolo');
        const result = await response.json();
        console.log('API response:', result);
        if (result.error) {
            console.error('Error:', result.error);
            return;
        }

        const chatMessages = document.getElementById('chat-messages');
        const aiMessage = document.createElement('div');
        aiMessage.className = 'message ai';
        
        if (result.response) {
            aiMessage.innerHTML = result.response;
        }

        if (result.tts_url) {
            handleAudioPlayback(result.tts_url, aiMessage);
        } else {
            chatMessages.appendChild(aiMessage);
        }

    } catch (error) {
        console.error('Error:', error);
    }
});

document.getElementById('send-button').addEventListener('click', async () => {
    const sendButton = document.getElementById('send-button');
    const userInput = document.getElementById('chat-input').value;
    if (!userInput) return;

    sendButton.disabled = true;
    sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

    const chatMessages = document.getElementById('chat-messages');
    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.textContent = userInput;
    chatMessages.appendChild(userMessage);

    try {
        const response = await fetch('/api/gemini', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_input: userInput })
        });
        const result = await response.json();
        console.log('API response:', result);

        const aiMessage = document.createElement('div');
        aiMessage.className = 'message ai';

        const youtubeInfo = extractYouTubeInfo(result.response);
        if (youtubeInfo) {
            // Nếu có video YouTube, chỉ hiển thị video
            const videoContainer = document.createElement('div');
            videoContainer.className = 'video-container';

            const videoTitle = document.createElement('div');
            videoTitle.className = 'video-title';
            videoTitle.innerHTML = `${youtubeInfo.title}`;
            videoContainer.appendChild(videoTitle);

            const iframe = document.createElement('iframe');
            iframe.src = `https://www.youtube.com/embed/${youtubeInfo.videoId}?autoplay=1&rel=0`;
            iframe.frameBorder = '0';
            iframe.allowFullscreen = true;
            iframe.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture';
            iframe.style.borderRadius = '2%';
            iframe.style.width = '100%';
            iframe.style.height = '315px';
            videoContainer.appendChild(iframe);

            aiMessage.appendChild(videoContainer);
            chatMessages.appendChild(aiMessage);

        } else {
            // Nếu không có video, hiển thị văn bản AI bình thường
            aiMessage.innerHTML = result.response || 'No response';
            chatMessages.appendChild(aiMessage); 
        }

        // Xử lý TTS nếu có
        if (result.tts_url) {
            handleAudioPlayback(result.tts_url, aiMessage);
        }

        chatMessages.scrollTop = chatMessages.scrollHeight;

    } catch (error) {
        console.error('Error:', error);
    } finally {
        sendButton.disabled = false;
        sendButton.innerHTML = 'Send';
    }


    document.getElementById('chat-input').value = '';
});

document.getElementById('chat-input').addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault();
        document.getElementById('send-button').click();
    }
});

// Unified microphone button functionality
const microphoneButton = document.getElementById('microphone');

microphoneButton.addEventListener('click', async function () {
    const icon = this.querySelector('i');
    const chatInput = document.getElementById('chat-input'); // Lấy phần tử input
    this.classList.toggle('active');

    if (icon.classList.contains('fa-microphone')) {
        icon.classList.remove('fa-microphone');
        icon.classList.add('fa-stop');

        
        // Start recording
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        const audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            // Đổi placeholder thành "Chờ phản hồi Speech to Text" khi dừng ghi âm
            

            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'audio.wav');

            try {
                const response = await fetch('/api/audio', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.text) {
                    const chatInput = document.getElementById('chat-input');
                    chatInput.value = result.text; // Insert transcribed text into chat input

                    const sendButton = document.getElementById('send-button');
                    sendButton.click(); // Kích hoạt sự kiện click của nút gửi
                } else {
                    console.error('Transcription failed:', result.error || 'Unknown error');
                }
            } catch (error) {
                console.error('Error sending audio:', error);
            } finally {
                // Đổi placeholder trở lại giá trị mặc định sau khi nhận phản hồi từ API
                chatInput.placeholder = "Nhập tin nhắn của bạn...";
            }
        };

        mediaRecorder.start();

        // Stop recording after 5 seconds (or when the button is clicked again)
        setTimeout(() => {
            if (mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                stream.getTracks().forEach(track => track.stop());
                icon.classList.remove('fa-stop');
                icon.classList.add('fa-microphone');
            }
        }, 5000);

    } else {
        // Stop recording immediately if button is clicked again
        icon.classList.remove('fa-stop');
        icon.classList.add('fa-microphone');
        chatInput.placeholder = "Chờ phản hồi Speech to Text";
    }
});
