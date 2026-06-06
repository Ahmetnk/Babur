import sounddevice as sd
from pathlib import Path
from scipy.io.wavfile import write

def record_audio(
    filename: str = "data/input.wav",
    duration: int = 8,
    sample_rate: int = 16000,
    gain: float = 4.0
) -> str:
    """
    Records audio from microphone and saves it as a WAV file.
    """

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Recording will start in 1 second...")
    sd.sleep(1000)
    print(f"Recording for {duration} seconds. Speak now.")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    write(output_path, sample_rate, audio)

    print(f"Saved audio to: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    record_audio(duration=8)