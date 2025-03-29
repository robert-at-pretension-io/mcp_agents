I'll create some examples of how to use `StreamedAudioInput` in a Python loop with user interaction. These examples will show how to capture audio from a user's microphone and process it through the OpenAI Agents SDK voice pipeline.

### Example 1: Basic Microphone Capture Loop

This example shows how to continuously capture audio from a microphone and feed it to a voice agent:

```python
import asyncio
import numpy as np
import sounddevice as sd
from agents import Agent, function_tool
from agents.voice import (
    StreamedAudioInput,
    SingleAgentVoiceWorkflow, 
    VoicePipeline
)

# Create a simple agent with a tool
@function_tool
def get_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

# Create the agent
agent = Agent(
    name="Voice Assistant",
    instructions="You're a helpful voice assistant. Keep responses brief and conversational.",
    model="gpt-4o-mini",
    tools=[get_time],
)

# Create the voice pipeline
pipeline = VoicePipeline(
    workflow=SingleAgentVoiceWorkflow(agent)
)

# Audio parameters
SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
CHANNELS = 1
DTYPE = np.int16

# Flag to control the loop
recording = True

# Function to process microphone input
async def process_microphone_input():
    # Create streamed audio input
    audio_input = StreamedAudioInput()
    
    # Start the pipeline
    result = await pipeline.run(audio_input)
    
    # Create a task to handle the output
    asyncio.create_task(process_output(result))
    
    # Set up microphone input callback
    def audio_callback(indata, frames, time, status):
        # Convert the input data to the right format
        audio_data = indata[:, 0].astype(DTYPE)
        # Push audio to the input stream
        asyncio.create_task(audio_input.add_audio(audio_data))
    
    # Start the microphone stream
    with sd.InputStream(
        callback=audio_callback,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype=DTYPE
    ):
        print("Listening... (Press Ctrl+C to stop)")
        # Keep the stream running until stopped
        while recording:
            await asyncio.sleep(0.1)

# Function to process pipeline output
async def process_output(result):
    # Create an audio player
    player = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=DTYPE
    )
    player.start()
    
    # Process the streaming result
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            # Play audio response
            player.write(event.data)
        elif event.type == "voice_stream_event_lifecycle":
            if event.event == "turn_started":
                print("Assistant is speaking...")
            elif event.event == "turn_ended":
                print("Assistant finished speaking.")
        elif event.type == "voice_stream_event_error":
            print(f"Error: {event.error}")
    
    player.stop()

# Main function to run the application
async def main():
    global recording
    try:
        await process_microphone_input()
    except KeyboardInterrupt:
        recording = False
        print("\nStopping...")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Push-to-Talk Interface

This example implements a push-to-talk system where the user explicitly controls when they're speaking:

```python
import asyncio
import numpy as np
import sounddevice as sd
import keyboard  # You'll need to install this with pip
from agents import Agent
from agents.voice import (
    StreamedAudioInput,
    SingleAgentVoiceWorkflow, 
    VoicePipeline
)

# Create a simple agent
agent = Agent(
    name="Voice Assistant",
    instructions=(
        "You are a helpful voice assistant. Be concise and conversational. "
        "The user will press a key when they want to speak."
    ),
    model="gpt-4o-mini",
)

# Audio parameters
SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
CHANNELS = 1
DTYPE = np.int16

async def main():
    # Create the voice pipeline
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent)
    )
    
    # Create streamed audio input
    audio_input = StreamedAudioInput()
    
    # Start the pipeline
    result = await pipeline.run(audio_input)
    
    # Setup audio output
    player = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=DTYPE
    )
    player.start()
    
    # Start output processor
    async def process_output():
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                player.write(event.data)
            elif event.type == "voice_stream_event_lifecycle":
                if event.event == "turn_started":
                    print("Assistant is speaking...")
                elif event.event == "turn_ended":
                    print("Assistant finished speaking. Press SPACE when you want to speak.")
    
    # Start the output processor as a task
    asyncio.create_task(process_output())
    
    # Initialize microphone stream (but don't start it yet)
    audio_stream = sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype=DTYPE
    )
    
    print("Press and hold SPACE to speak. Release to stop speaking.")
    
    # Main interaction loop
    running = True
    while running:
        # Wait for user input (space key press/release)
        if keyboard.is_pressed('space'):
            if not audio_stream.active:
                print("Listening...")
                audio_stream.start()
                
            # Capture audio while space is pressed
            indata, _ = audio_stream.read(CHUNK_SIZE)
            audio_data = indata[:, 0].astype(DTYPE)
            await audio_input.add_audio(audio_data)
        else:
            # If space was released and stream is active, stop the stream
            if audio_stream.active:
                audio_stream.stop()
                print("Processing...")
        
        # Check for exit command (Esc key)
        if keyboard.is_pressed('esc'):
            print("Exiting...")
            running = False
        
        # Short sleep to prevent CPU hogging
        await asyncio.sleep(0.01)
    
    # Clean up
    if audio_stream.active:
        audio_stream.stop()
    player.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 3: Voice-Activated Conversation with VAD (Voice Activity Detection)

This example uses a simple energy-based voice activity detection to start and stop recording automatically:

```python
import asyncio
import numpy as np
import sounddevice as sd
from agents import Agent
from agents.voice import (
    StreamedAudioInput,
    SingleAgentVoiceWorkflow, 
    VoicePipeline
)

# Create a simple agent
agent = Agent(
    name="Conversation Assistant",
    instructions=(
        "You're a conversation partner who responds naturally. "
        "Keep responses concise and engaging."
    ),
    model="gpt-4o-mini",
)

# Audio parameters
SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = np.int16
CHUNK_SIZE = 1024

# VAD parameters
ENERGY_THRESHOLD = 500  # Adjust based on your microphone and environment
SILENCE_LIMIT = 1.5  # seconds of silence before stopping capture

async def main():
    # Create the pipeline with StreamedAudioInput
    audio_input = StreamedAudioInput()
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
    result = await pipeline.run(audio_input)
    
    # Set up audio output
    player = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype=DTYPE)
    player.start()
    
    # Process output in background
    async def handle_output():
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                player.write(event.data)
            elif event.type == "voice_stream_event_lifecycle":
                if event.event == "turn_started":
                    print("Assistant is speaking...")
                    # Pause microphone input while assistant is speaking
                    # (In a real app, you'd want to manage this)
                elif event.event == "turn_ended":
                    print("Assistant finished speaking. I'm listening...")
    
    # Start output handler
    asyncio.create_task(handle_output())
    
    # Set up microphone with VAD
    silence_counter = 0
    is_speaking = False
    
    def audio_callback(indata, frames, time, status):
        nonlocal silence_counter, is_speaking
        
        # Calculate energy level (simple approach)
        energy = np.sqrt(np.mean(indata**2)) * 1000
        
        # Voice activity detection
        if energy > ENERGY_THRESHOLD:
            if not is_speaking:
                print("Voice detected!")
                is_speaking = True
            silence_counter = 0
        else:
            if is_speaking:
                silence_counter += frames / SAMPLE_RATE
                if silence_counter >= SILENCE_LIMIT:
                    print("Silence detected, processing...")
                    is_speaking = False
                    silence_counter = 0
        
        # Only send audio when user is speaking
        if is_speaking:
            audio_data = indata[:, 0].astype(DTYPE)
            asyncio.create_task(audio_input.add_audio(audio_data))
    
    # Start audio stream
    print("Listening for voice input...")
    with sd.InputStream(
        callback=audio_callback,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        dtype=DTYPE
    ):
        try:
            # Run indefinitely until interrupted
            while True:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping application...")
    
    player.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 4: Continuous Conversation with Interrupt Support

This example demonstrates a more sophisticated approach that allows for continuous conversation with the ability to interrupt the assistant:

```python
import asyncio
import numpy as np
import sounddevice as sd
from agents import Agent
from agents.voice import (
    StreamedAudioInput,
    SingleAgentVoiceWorkflow, 
    VoicePipeline,
    VoicePipelineConfig
)

# Create a conversation agent
agent = Agent(
    name="Continuous Conversation Bot",
    instructions=(
        "You're a chatbot designed for continuous, natural conversation. "
        "Keep responses concise and engaging. If interrupted, adapt naturally."
    ),
    model="gpt-4o-mini",
)

# Audio parameters
SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = np.int16
CHUNK_SIZE = 1024

# Voice detection parameters
ENERGY_THRESHOLD = 500
SILENCE_DURATION = 1.0  # seconds

async def main():
    # Create custom pipeline config
    config = VoicePipelineConfig(
        tts_voice="nova",  # Using a specific voice
        workflow_name="continuous_conversation"
    )
    
    # Set up the voice pipeline
    audio_input = StreamedAudioInput()
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent),
        config=config
    )
    result = await pipeline.run(audio_input)
    
    # State variables
    assistant_speaking = False
    user_speaking = False
    silence_frames = 0
    interrupted = False
    
    # Setup audio player
    player = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype=DTYPE)
    player.start()
    
    # Audio queue for the assistant's output
    audio_queue = asyncio.Queue()
    
    # Process assistant's output
    async def process_output():
        nonlocal assistant_speaking
        
        async for event in result.stream():
            if event.type == "voice_stream_event_lifecycle":
                if event.event == "turn_started":
                    print("Assistant is speaking...")
                    assistant_speaking = True
                elif event.event == "turn_ended":
                    print("Assistant finished.")
                    assistant_speaking = False
            
            elif event.type == "voice_stream_event_audio":
                # If interrupted by user, discard audio
                if not interrupted:
                    await audio_queue.put(event.data)
                else:
                    print("User interrupted - discarding assistant audio")
            
            elif event.type == "voice_stream_event_error":
                print(f"Error: {event.error}")
    
    # Play assistant's audio
    async def play_audio():
        nonlocal interrupted
        
        while True:
            # Get next audio chunk
            try:
                audio_chunk = await audio_queue.get()
                
                # Skip if interrupted
                if interrupted:
                    interrupted = False
                    continue
                
                # Play the audio
                player.write(audio_chunk)
                audio_queue.task_done()
            except Exception as e:
                print(f"Audio playback error: {e}")
            
            await asyncio.sleep(0.01)
    
    # Start output processors
    asyncio.create_task(process_output())
    asyncio.create_task(play_audio())
    
    # Process microphone input
    def audio_callback(indata, frames, time, status):
        nonlocal user_speaking, silence_frames, interrupted, assistant_speaking
        
        # Calculate energy level
        audio_data = indata[:, 0].astype(DTYPE)
        energy = np.sqrt(np.mean(audio_data**2)) * 1000
        
        # Voice activity detection
        if energy > ENERGY_THRESHOLD:
            # User is speaking
            if not user_speaking:
                print("User started speaking")
                user_speaking = True
                
                # Check if we need to interrupt the assistant
                if assistant_speaking:
                    print("Interrupting assistant...")
                    interrupted = True
                    
            # Reset silence counter
            silence_frames = 0
            
            # Add audio to input
            asyncio.create_task(audio_input.add_audio(audio_data))
        else:
            # No voice detected
            if user_speaking:
                silence_frames += frames
                
                # Check if silence has persisted long enough
                if silence_frames >= SILENCE_DURATION * SAMPLE_RATE:
                    print("User stopped speaking")
                    user_speaking = False
                    silence_frames = 0
    
    # Start microphone
    print("Starting conversation... Speak naturally.")
    with sd.InputStream(
        callback=audio_callback,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        dtype=DTYPE
    ):
        try:
            # Run until interrupted
            while True:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\nEnding conversation...")
    
    player.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Notes on the Examples

1. **Environment Setup**: These examples require the following packages:
   - `openai-agents` (with voice support)
   - `numpy`
   - `sounddevice` for audio capture and playback
   - `keyboard` (for the push-to-talk example)

2. **Audio Parameters**: The default sample rate for the OpenAI Agents SDK is 24kHz, so these examples use that. You may need to adjust audio settings based on your microphone and hardware.

3. **Voice Activity Detection (VAD)**: The energy-based VAD used in examples 3 and 4 is a simple approach. For better results in noisy environments, consider using more sophisticated VAD libraries like `webrtcvad`.

4. **Error Handling**: These examples include basic error handling, but in a production environment, you'd want more robust error recovery mechanisms.

5. **Customization**: You can customize the agent's instructions, model, and voice based on your requirements. The OpenAI Agents SDK supports different voices and languages.

These examples demonstrate different interaction patterns with `StreamedAudioInput`, from basic continuous recording to more sophisticated turn-taking models with voice activity detection.