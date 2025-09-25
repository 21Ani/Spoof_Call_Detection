from flask import Flask, render_template, request, jsonify
from speech_emotion_recognition import speechEmotionRecognition
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Audio Emotion Analysis
def audio_dash(file_path):
    model_sub_dir = r'audio.hdf5'
    SER = speechEmotionRecognition(model_sub_dir)

    step = 1  # in sec
    sample_rate = 16000  # in Hz
    emotions, timestamp = SER.predict_emotion_from_file(file_path, chunk_step=step * sample_rate)

    # distribution of emotions
    emotion_dist = [(emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
    return emotion_dist

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith('.wav'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Run emotion analysis
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        emo_tot = audio_dash(file_path)

        # Plot emotion distribution
        emo_plot = [emo * 100 for emo in emo_tot]
        plt.bar(emotions, emo_plot, color='red')
        plt.xlabel('Emotions')
        plt.ylabel('Percentage')
        plt.title('Emotion Distribution in Uploaded Audio')
        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'emotion_dist.png')
        plt.savefig(plot_path)
        plt.close()

        # Compute spoofness
        emo_wt = np.array([-0.3, 0, -1, 0.8, 0.5, -0.5, 0])
        spoofness = np.dot(emo_wt, emo_tot)
        spoofness = (spoofness + 1) / 2  # Normalize 0-1

        return jsonify({
            'message': 'File processed successfully.',
            'spoofness': float(spoofness),
            'plot': plot_path
        })

    return jsonify({'error': 'Invalid file format. Only .wav allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
