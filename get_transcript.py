# import whisper
# import os
#
# model = whisper.load_model("small", device = "cpu")  # "tiny", "base", "small", "medium", or "large"
# for (dirpath, dirnames, filenames) in os.walk("./videos/"):
#     files = filenames
# dirpath = "./videos/"
# if not files:
#     print("Could not load video paths...")
#     exit(0)
# for i, filename in enumerate(files):
#     print(f"{i}. 🌐{filename} will be transcribed now1")
#     result = model.transcribe(f"{dirpath}/{filename}")
#
#     with open(f"./data/{filename}", "a") as f:
#         f.write(result["text"])
#     print(f"✅ Saved {filename} to {dirpath}...")
# print("Transcription complete!")
#

import whisper
import ffmpeg
import os

def convert_audio(input_file, output_file="temp.wav"):
    ffmpeg.input(input_file).output(output_file, ar=16000, ac=1).run(overwrite_output=True)
    return output_file

model = whisper.load_model("small", device="cpu")  # Use "tiny" or "base"

for (dirpath, dirnames, filenames) in os.walk("./videos/"):
    files = filenames

for i, filename in enumerate(files[131:]):
    if not filename.endswith((".mp3", ".mp4", ".wav", ".m4a")):
        continue
    
    print(f"{i+91}.🔍Transcribing: {filename}...")
    audio_path = convert_audio(f"./videos/{filename}")
    
    result = model.transcribe(audio_path, temperature=0)  # Faster transcription
    
    with open(f"./data/{filename}.txt", "w") as f:
        f.write(result["text"])

    os.remove(audio_path)  # Cleanup temp file
    print(f"✅ Done: {filename}")

print("Transcription complete!")

