from audio.recorder import record_audio
from audio.stt import transcribe_audio


def main():
    audio_file = record_audio(duration=5)

    text = transcribe_audio(audio_file)

    print("\nYou said:")
    print(text)


if __name__ == "__main__":
    main()