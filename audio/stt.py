from faster_whisper import WhisperModel

model = WhisperModel(
        "medium",
        device="cpu",
        compute_type="int8"
    )

def transcribe_audio(
    filename: str = "data/input.wav",
    model_size: str = "medium",
    language: str = "tr"
) -> str:
    """
    Converts a WAV audio file into text using Faster-Whisper.
    """

    segments, info = model.transcribe(
        filename,
        language=language,
        beam_size=5
    )

    text_parts = []

    for segment in segments:
        text_parts.append(segment.text)

    text = " ".join(text_parts).strip()

    print("Detected language:", info.language)
    print("Transcription:", text)

    return text


if __name__ == "__main__":
    transcribe_audio("data/input.wav")