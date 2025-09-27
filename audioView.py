import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide the main tkinter window
Tk().withdraw()

# Open file dialog to select audio file
filename = askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac")])

if not filename:
    print("No file selected, exiting.")
    exit()

# Load audio with pydub
audio = AudioSegment.from_file(filename)


samples = np.array(audio.get_array_of_samples())
if audio.channels == 2:
    samples = samples.reshape((-1, 2))
    samples = samples.mean(axis=1)  # stereo to mono

rate = audio.frame_rate

# Plot spectrogram
plt.specgram(samples, Fs=rate, NFFT=1024, noverlap=512, cmap="gray_r")
plt.title(f"Spectrogram of {filename.split('/')[-1]}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()
