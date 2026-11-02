# import yt_dlp
# from pydub import AudioSegment
# import os

# DOWNLOAD_DIR = 'downloads'

# os.makedirs(DOWNLOAD_DIR,exist_ok = True)

# def download_youtube_audio(url :str) ->str:
#     output_path = os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s')
#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'outtmpl': output_path,
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'wav',
#             'preferredquality': '192',
#         }],
#         'quiet': True,
#     }
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info  = ydl.extract_info(url, download=True)
#         filename = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
#     return filename



# def convert_to_wav(input_path: str) -> str:
# # Convert any audio/video file to WAV format using pydub.
#     output_path = os.path.splitext(input_path)[0] + '_converted.wav'
#     audio = AudioSegment.from_file(input_path)
#     audio = audio.set_channels(1).set_frame_rate(16000)  # Convert to mono and set frame rate
#     audio.export(output_path, format='wav')
#     return output_path


# def chunk_audio(wav_path : str,chunk_minutes : int = 10) -> list:
#     audio = AudioSegment.from_wav(wav_path)
#     chunk_length_ms = chunk_minutes * 60 * 1000  # Convert minutes to milliseconds
#     chunks = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         chunk_filename = f"{os.path.splitext(wav_path)[0]}_chunk_{i // chunk_length_ms}.wav"
#         chunk.export(chunk_filename, format='wav')
#         chunks.append(chunk_filename)
#     return chunks


# def process_input(source : str) -> list:
#     if source.startswith('http://') or source.startswith('https://'):
#         # It's a YouTube URL
#         downloaded_file = download_youtube_audio(source)
#         wav_file = convert_to_wav(downloaded_file)
#     else:
#         # It's a local file path
#         wav_file = convert_to_wav(source)

#     chunks = chunk_audio(wav_file)
#     return chunks




import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            filename = os.path.join(
                DOWNLOAD_DIR,
                f"{info['id']}.wav"
            )

            if not os.path.exists(filename):
                raise FileNotFoundError(
                    "Downloaded WAV file not found."
                )

            return filename

    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(
            "YouTube download failed. YouTube may be "
            "blocking the hosting server (HTTP 403). "
            "Try uploading the video/audio file instead."
        ) from e


def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_length_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]

        chunk_filename = (
            f"{os.path.splitext(wav_path)[0]}_chunk_"
            f"{i // chunk_length_ms}.wav"
        )

        chunk.export(chunk_filename, format="wav")
        chunks.append(chunk_filename)

    return chunks


def process_input(source: str) -> list:
    if source.startswith(("http://", "https://")):
        downloaded_file = download_youtube_audio(source)
        wav_file = convert_to_wav(downloaded_file)
    else:
        wav_file = convert_to_wav(source)

    return chunk_audio(wav_file)
