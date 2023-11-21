from keras.models import load_model
from keras.models import model_from_json
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils import shuffle
from keras.utils import np_utils
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import pyaudio
import wave
import numpy
import sounddevice as sd
from scipy.io.wavfile import write
import glob
import easygui


mylist= os.listdir('data/')  



feeling_list=[]
for item in mylist:
    if item[6:-16]=='02' and int(item[18:-4])%2==0:
        feeling_list.append('female_calm')
    elif item[6:-16]=='02' and int(item[18:-4])%2==1:
        feeling_list.append('male_calm')
    elif item[6:-16]=='03' and int(item[18:-4])%2==0:
        feeling_list.append('female_happy')
    elif item[6:-16]=='03' and int(item[18:-4])%2==1:
        feeling_list.append('male_happy')
    elif item[6:-16]=='04' and int(item[18:-4])%2==0:
        feeling_list.append('female_sad')
    elif item[6:-16]=='04' and int(item[18:-4])%2==1:
        feeling_list.append('male_sad')
    elif item[6:-16]=='05' and int(item[18:-4])%2==0:
        feeling_list.append('female_angry')
    elif item[6:-16]=='05' and int(item[18:-4])%2==1:
        feeling_list.append('male_angry')
    elif item[6:-16]=='06' and int(item[18:-4])%2==0:
        feeling_list.append('female_fearful')
    elif item[6:-16]=='06' and int(item[18:-4])%2==1:
        feeling_list.append('male_fearful')
    elif item[:1]=='a':
        feeling_list.append('male_angry')
    elif item[:1]=='f':
        feeling_list.append('male_fearful')
    elif item[:1]=='h':
        feeling_list.append('male_happy')
    #elif item[:1]=='n':
        #feeling_list.append('neutral')
    elif item[:2]=='sa':
        feeling_list.append('male_sad')

labels = pd.DataFrame(feeling_list)

df = pd.DataFrame(columns=['feature'])
df3 = pd.DataFrame(df['feature'].values.tolist())
newdf = pd.concat([df3,labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})
rnewdf = shuffle(newdf)

rnewdf=rnewdf.fillna(0)

newdf1 = np.random.rand(len(rnewdf)) < 0.8
train = rnewdf[newdf1]
test = rnewdf[~newdf1]

trainfeatures = train.iloc[:, :-1]
trainlabel = train.iloc[:, -1:]
testfeatures = test.iloc[:, :-1]
testlabel = test.iloc[:, -1:]

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))









json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")

def predict():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    data, sampling_rate = librosa.load('output10.wav')
    X, sample_rate = librosa.load('output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive

    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T

    twodim= np.expand_dims(livedf2, axis=2)

    livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)

    livepreds1=livepreds.argmax(axis=1)

    liveabc = livepreds1.astype(int).flatten()

    livepredictions = (lb.inverse_transform((liveabc)))
    sign=livepredictions

    if sign == 'male_sad' or sign=='female_sad':
        easygui.msgbox("It's OK Everything will be fine", title="Notification")
    elif sign == 'male_angry' or sign=='female_angry' :
        easygui.msgbox("Relax...................", title="Notification")
    elif sign == 'male_fearful' or sign=='female_fearful':
        easygui.msgbox("ur very angry Relax....Take a deep breath", title="Notification")
    elif sign == 'male_angry' or sign == 'female_angry' :
        easygui.msgbox("Keep Calm......", title="Notification")
    else:
        sign = 'neutral'
        easygui.msgbox("Emotion detected is neutral", title="Notification")


def record_audio():
    CHUNK = 1024 
    FORMAT = pyaudio.paInt16 #paInt8
    CHANNELS = 2 
    RATE = 44100 #sample rate
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "output10.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


top = tk.Tk()
top.geometry('600x500')
top.title('Speech based Emotion Detection ')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

upload = Button(top, text="Record Audio", command=record_audio, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload.place(x=140,y=200)

upload1 = Button(top, text="Emotion", command=predict, padx=25, pady=6)
upload1.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload1.place(x=350,y=200)


heading = Label(top, text="Speech based Emotion Detection", pady=20, font=('arial', 15, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()




