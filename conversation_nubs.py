from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import speech_recognition as sr
import pyttsx3

# Function to capture audio input from the microphone
def capture_audio():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Say something...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio. Please try again.")
            return None

# Function to speak the AI response with a specific accent
def speak_response(response_text, accent="english"):
    engine = pyttsx3.init()

    # Configure the voice with the desired accent
    voices = engine.getProperty('voices')
    for voice in voices:
        if accent.lower() in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    engine.say(response_text)
    engine.runAndWait()

# Download and setup the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
# Main loop
while True:
    # Capture audio input from the microphone
    utterance = capture_audio()

    # Check if the recognition failed
    if utterance is None:
        continue

    # Check if the user wants to exit
    if utterance.lower() == "exit":
        print("Exiting...")
        break

    # Tokenize the utterance
    inputs = tokenizer(utterance, return_tensors="pt")

    # Passing through the utterance to the Blenderbot model
    res = model.generate(**inputs)

    # Decoding the model output
    output_text = tokenizer.decode(res[0])

    # Print the input and output
    print("Input:", utterance)
    print("Output:", output_text)

    # Speak the AI response with a specific accent (e.g., British English)
    speak_response(output_text, accent="english_gb")