import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from scipy.io.wavfile import write
import os
import subprocess
import sys

DURATION = 5  # seconds
SAMPLE_RATE = 44100  # samples per second

class DrawingApp:
    def __init__(self, master):
        self.master = master

        # Initial canvas size
        self.img_width = 512
        self.img_height = 512

        # Create a resizable canvas, packed with expand/stretch
        self.canvas = tk.Canvas(master, width=self.img_width, height=self.img_height, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Create image and draw for the canvas backing
        self.image = Image.new("L", (self.img_width, self.img_height), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None

        # Bind mouse drawing events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

        # Bind resizing event
        self.canvas.bind('<Configure>', self.resize)

        self.save_button = tk.Button(master, text="Save & Generate Audio", command=self.process)
        self.save_button.pack()

    def paint(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill='white', width=3)
            self.draw.line((self.last_x, self.last_y, event.x, event.y), fill=255, width=3)
        self.last_x, self.last_y = event.x, event.y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def resize(self, event):
        # Save old image
        old_image = self.image

        # Update size attributes
        self.img_width = event.width
        self.img_height = event.height

        # Create new image with updated size
        self.image = Image.new("L", (self.img_width, self.img_height), color=0)

        # Paste the old drawing resized to new size (to keep drawing on resize)
        old_resized = old_image.resize((self.img_width, self.img_height), Image.LANCZOS)
        self.image.paste(old_resized)

        # Create new draw object
        self.draw = ImageDraw.Draw(self.image)

        # Clear the canvas and redraw the resized image on canvas
        self.canvas.delete("all")
        self.tk_img = tk.PhotoImage(master=self.canvas, width=self.img_width, height=self.img_height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # Draw the resized image onto canvas using tkinter-compatible PhotoImage
        # We need to convert the grayscale PIL image to a format compatible with PhotoImage
        self.update_canvas_image()

    def update_canvas_image(self):
        # Convert PIL image (L mode) to PGM byte string for PhotoImage
        # This is a workaround to display grayscale image on tkinter canvas
        img8 = self.image.convert("L")
        data = img8.tobytes("raw", "L")
        self.tk_img.put(data)

    def process(self):
        # Save drawing to PNG
        save_path = "drawing.png"
        self.image.save(save_path)
        print("Saved drawing. Now generating audio...")
        generate_audio_from_image(save_path, self.img_width, self.img_height)

def generate_audio_from_image(image_path, img_width, img_height):
    img = Image.open(image_path).convert("L")
    img = img.resize((img_width, img_height))
    img_data = np.array(img).astype(float)
    img_data = img_data[::-1]  # Flip vertically so high freqs are top

    img_data /= 255.0

    n_samples = SAMPLE_RATE * DURATION
    n_fft = img_height * 2
    hop_size = n_samples // img_width

    audio = np.zeros(n_samples)
    window = np.hanning(n_fft)

    for t in range(img_width):
        spectrum = img_data[:, t]
        phase = np.random.rand(len(spectrum)) * 2 * np.pi
        real = spectrum * np.cos(phase)
        imag = spectrum * np.sin(phase)
        freqs = real + 1j * imag

        spectrum_full = np.concatenate([freqs, np.conj(freqs[::-1])])

        frame = np.fft.ifft(spectrum_full).real
        frame *= window

        start = t * hop_size
        end = start + n_fft
        if end <= len(audio):
            audio[start:end] += frame[:min(len(frame), len(audio)-start)]

    audio /= np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)

    output_path = os.path.abspath("output.wav")
    write(output_path, SAMPLE_RATE, audio)
    print(f"Audio saved to {output_path}")

    if sys.platform == "win32":
        subprocess.run(f'explorer /select,"{output_path}"')

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Draw Something to Encode into Audio Spectrogram")
    app = DrawingApp(root)
    root.mainloop()
