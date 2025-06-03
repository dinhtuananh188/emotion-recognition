// Browser camera setup
let videoStream = null;
let videoElement = null;
let canvasElement = null;
let canvasContext = null;
let yoloLabels = []; // Array to store YOLO labels

// Function to log YOLO labels
function logYoloLabels() {
    console.log('YOLO Labels in first 5 seconds:', yoloLabels);
    
    // Count occurrences of each emotion
    const emotionCount = {};
    yoloLabels.forEach(emotion => {
        emotionCount[emotion] = (emotionCount[emotion] || 0) + 1;
    });
    
    // Find the most frequent emotion
    let maxCount = 0;
    let mostFrequentEmotion = '';
    
    for (const [emotion, count] of Object.entries(emotionCount)) {
        if (count > maxCount) {
            maxCount = count;
            mostFrequentEmotion = emotion;
        }
    }
    
    console.log('Thống kê cảm xúc:');
    console.log(emotionCount);
    console.log(`Cảm xúc xuất hiện nhiều nhất: ${mostFrequentEmotion} (${maxCount} lần)`);

    // Only trigger AI response, do not append user message manually
    const chatInput = document.getElementById('chat-input');
    chatInput.value = `Tôi đang cảm thấy ${mostFrequentEmotion}`;
    document.getElementById('send-button').click();
}

async function initializeBrowserCamera() {
    try {
        videoElement = document.getElementById('browser-camera');
        canvasElement = document.getElementById('camera-canvas');
        canvasContext = canvasElement.getContext('2d');

        videoStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        videoElement.srcObject = videoStream;
        
        // Set canvas size to match video
        videoElement.addEventListener('loadedmetadata', () => {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
        });

        return true;
    } catch (error) {
        console.error('Error accessing camera:', error);
        handleCameraError();
        return false;
    }
}

// Function to capture frame from browser camera
function captureBrowserFrame() {
    if (!videoElement || !canvasElement || !canvasContext) return null;
    
    canvasContext.drawImage(videoElement, 0, 0);
    return canvasElement.toDataURL('image/jpeg', 0.8);
}

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

    // Khi phát xong thì đổi icon về ▶
    audio.addEventListener('ended', () => {
        playPauseButton.innerHTML = '▶';
        isPlaying = false;
    });

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

// Function to process frame with YOLO
let isProcessing = false;
async function processFrameWithYOLO() {
    if (!videoElement || !canvasElement || !canvasContext || isProcessing) return;
    
    isProcessing = true;

    try {
        const frameData = captureBrowserFrame();
        if (!frameData) {
            isProcessing = false;
            return;
        }

        const response = await fetch('/api/yolo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ frame: frameData })
        });

        const result = await response.json();
        if (result.error) {
            console.error('Error processing frame:', result.error);
            // Keep processing even if there's an error with this frame
            return;
        }

        // Store YOLO label if available
        if (result.label) {
            yoloLabels.push(result.label);
        }

        // Update video element with processed frame
        const img = new Image();
        img.onload = () => {
            // Draw the processed frame with bounding boxes
            canvasContext.drawImage(img, 0, 0);
            // Display the canvas instead of the video
            videoElement.style.display = 'none';
            canvasElement.style.display = 'block';
        };
        img.src = 'data:image/jpeg;base64,' + result.processed_frame;

    } catch (error) {
        console.error('Error processing frame:', error);
    } finally {
        isProcessing = false;
        // Process the next frame after a short delay
        setTimeout(processFrameWithYOLO, 100); 
    }
}

// Start processing frames
let processingInterval = null;

function startFrameProcessing() {
    // Stop the interval if it was running
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
    // Start the processing loop
    processFrameWithYOLO();
}

function stopFrameProcessing() {
    // This function is not used in the new loop, but keep it for completeness
    // if the interval was ever used elsewhere.
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
    // No need to stop setTimeout loops explicitly unless you add a flag
}

// Update DOMContentLoaded to start frame processing
document.addEventListener('DOMContentLoaded', async () => {
    const success = await initializeBrowserCamera();
    if (success) {
        startFrameProcessing();
        
        // Set timer to log YOLO labels after 5 seconds
        setTimeout(() => {
            logYoloLabels();
            // Clear the labels array after logging
            yoloLabels = [];
        }, 5000);
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

// Update track_yolo function to use browser camera
async function trackYOLO() {
    try {
        const frameData = captureBrowserFrame();
        if (!frameData) return;

        const response = await fetch('/api/track_yolo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ frame: frameData })
        });

        const result = await response.json();
        console.log('API response:', result);

        const chatMessages = document.getElementById('chat-messages');
        const aiMessage = document.createElement('div');
        aiMessage.className = 'message ai';

        // Update the processed frame regardless of whether we got a label or not
        if (result.processed_frame) {
            const img = new Image();
            img.onload = () => {
                canvasContext.drawImage(img, 0, 0);
                videoElement.style.display = 'none';
                canvasElement.style.display = 'block';
            };
            img.src = 'data:image/jpeg;base64,' + result.processed_frame;
        }

        // Only show error message if there's an actual error
        if (result.error && !result.processed_frame) {
            console.error('Error:', result.error);
            aiMessage.innerHTML = 'Lỗi khi phát hiện khuôn mặt';
            chatMessages.appendChild(aiMessage);
            return;
        }

        // If we have a response, show it
        if (result.response) {
            aiMessage.innerHTML = result.response;
            chatMessages.appendChild(aiMessage);
        }

        // Handle TTS if available
        if (result.tts_url) {
            handleAudioPlayback(result.tts_url, aiMessage);
        }

    } catch (error) {
        console.error('Error:', error);
        const chatMessages = document.getElementById('chat-messages');
        const aiMessage = document.createElement('div');
        aiMessage.className = 'message ai';
        aiMessage.innerHTML = 'Lỗi khi phát hiện khuôn mặt';
        chatMessages.appendChild(aiMessage);
    }
}
