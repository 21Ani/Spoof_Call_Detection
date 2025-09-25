from flask import Flask, render_template, request, jsonify, send_from_directory
from speech_emotion_recognition import speechEmotionRecognition
import matplotlib.pyplot as plt
from pydub import AudioSegment
import time
import json
import numpy as np
import os

app = Flask(__name__)

# Voice Recording
def record(rec_duration, rec_sub_dir):
    # Instanciate new AudioRecorder object
    SER = speechEmotionRecognition()

    # Voice Recording
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    return


# Audio Emotion Analysis
def audio_dash(rec_sub_dir):
    # Sub dir to speech emotion recognition model
    model_sub_dir = r'audio.hdf5'

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)

    # Calculate emotion distribution
    emotion_dist = [(emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    return emotion_dist

# Add two emotion distributions
def add(emo1, emo2):
    return [emo1[i] + emo2[i] for i in range(0, len(emo1))]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result(): 
    return render_template('result.html')

@app.route('/video/<path:path>')
def serve_video(path):
    return send_from_directory('static', path) 

@app.route('/animations/<path:filename>')
def serve_animation(filename):
    return send_from_directory(os.path.join(app.root_path, 'static', 'animations'), filename)

@app.route('/start', methods=['POST'])
def start():
    rec_duration = 4 # Default Recording Duration
    on_call = True # Start Call
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    recording_chunk = 0 # Default Recording Chunk
    emo_tot = [0] * len(emotions) # Default Emotion Distribution

    while(True):
        with open('call.json', 'r') as f:
            on_call = int(json.load(f))
            f.close()
        if on_call == 0:
            break
        else:
            record(rec_duration, f"Recording\{recording_chunk}.wav")
            recording_chunk += 1

    with open('call.json', 'w') as f:
        f.write('1')
        f.close()
        
    for analyse_chunk in range(0, recording_chunk):
        emo_curr = audio_dash(f"Recording\{analyse_chunk}.wav")
        emo_tot = add(emo_tot, emo_curr)

    emo_avg = [emo / recording_chunk for emo in emo_tot]
    emo_plot = [emo * 100  for emo in emo_avg]
    plt.bar(emotions, emo_plot, color = 'red')
    plt.xlabel('Emotions')
    plt.ylabel('Percentage')
    plt.title('Emotion Distribution in Call')
    plt.savefig('emotion_dist.png')

    with open('emotion_dist.json', 'w') as f:
        f.write(str({emotions[i] : emo_avg[i] for i in range(0, len(emotions))}))
        f.close()

    # Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
    emo_wt = np.array([-0.3, 0, -1, 0.8, 0.5, -0.5, 0])
    spoofness = np.dot(emo_wt, emo_avg)
    spoofness += 1
    spoofness /= 2
    print(spoofness) 

    return jsonify({'message': 'Process started successfully.',"spoofness": spoofness})

@app.route('/stop', methods=['POST'])
def stop(): 
    with open('call.json', 'w') as f:
        json.dump(0, f)
        f.close()
    return jsonify({'message': 'Process stopped successfully.'})

if __name__ == '__main__':
    app.run(debug = True)