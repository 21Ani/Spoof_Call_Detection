from speech_emotion_recognition import speechEmotionRecognition
from pydub import AudioSegment
import time
import json
import numpy as np

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


if __name__ == '__main__':
    rec_duration = 10 # Default Recording Duration
    on_call = True # Start Call
    recording_chunk = 0 # Default Recording Chunk
    emo_tot = [0] * 7 # Default Emotion Distribution

    while(True):
        with open('call.json', 'r') as f:
            on_call = int(json.load(f))
            print(on_call, type(on_call))
            f.close()
        if on_call == 0:
            break
        else:
            record(rec_duration, f"Recording\{recording_chunk}.wav")
            recording_chunk += 1

    for analyse_chunk in range(0, recording_chunk):
        emo_curr = audio_dash(f"Recording\{analyse_chunk}.wav")
        emo_tot = add(emo_tot, emo_curr)

    emo_avg = [emo / recording_chunk for emo in emo_tot]
    with open('emoemotion_dist.json', 'w') as f:
        f.write({emotion : (emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()})
        f.close()

    with open('call.json', 'w') as f:
        f.write('1')
        f.close()

    # Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
    emo_wt = np.array([-0.3, 0, -1, 0.8, 0.5, -0.5, 0])
    spoofness = np.dot(emo_wt, emo_avg)
    print(spoofness)