import re
import assemblyai as aai

def adjust_temperature(temperature):
    print(f"Command: Temperature adjusted to {temperature} degrees.")

def schedule_heater(hours):
    print(f"Command: Heater will start working in {hours} hours.")

def schedule_heater_shutdown(hours):
    print(f"Command: Heater will stop working in {hours} hours.")

def parse_heater_commands(text):
    temp_pattern = r"I'd like to see the temperature cranked up to (\d+) degrees"
    schedule_pattern = r"open the heater in (\d+) hours"
    shutdown_pattern = r"shutdown the heater in (\d+) hours"

    temp_match = re.search(temp_pattern, text, re.IGNORECASE)
    if temp_match:
        adjust_temperature(temp_match.group(1))

    schedule_match = re.search(schedule_pattern, text, re.IGNORECASE)
    if schedule_match:
        schedule_heater(schedule_match.group(1))

    shutdown_match = re.search(shutdown_pattern, text, re.IGNORECASE)
    if shutdown_match:
        schedule_heater_shutdown(shutdown_match.group(1))

def transcribe_and_parse_commands(audio_path):
    aai.settings.api_key = "a4477fa34d284170aeb499608b711bdd"
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe("D:/作业/论文/ppt/Record (online-voice-recorder.com) 2.mp3")
    if transcript.status == 'completed':
        print('Result of speech to text:', transcript.text)
        parse_heater_commands(transcript.text)
    else:
        print("Transcription is not completed. Status:", transcript.status)


# Example usage
audio_path = "D:/作业/论文/ppt/Record (online-voice-recorder.com) 2.mp3"  # Use your actual audio file path
transcribe_and_parse_commands(audio_path)
