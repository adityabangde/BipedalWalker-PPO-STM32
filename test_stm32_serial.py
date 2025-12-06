import gymnasium as gym
import serial
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk

# Setup
SERIAL_PORT = 'COM5'
BAUD_RATE = 115200

# Connect serial
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
print(f"Connected to {SERIAL_PORT}")
time.sleep(2)
ser.reset_input_buffer()

# Make environment
env = gym.make('BipedalWalker-v3', render_mode='rgb_array')

# UI window
root = tk.Tk()
root.title("STM32 BipedalWalker Test")
canvas = tk.Canvas(root, width=600, height=400, bg='black')
canvas.pack()
status_label = tk.Label(root, text="Starting...", font=("Arial", 12))
status_label.pack()

# Variables
current_episode = 1
current_state = None
episode_reward = 0
current_step = 0

def start_episode():
    global current_state, episode_reward, current_step
    obs, info = env.reset()
    current_state = obs
    episode_reward = 0
    current_step = 0

def run_step():
    global current_episode, current_state, episode_reward, current_step
    
    # Check if we're done with all episodes
    if current_episode > 3:
        status_label.config(text="All episodes completed!")
        root.after(2000, close_everything)
        return
    
    # Check if episode should end
    if current_step >= 1600:
        print(f"Episode {current_episode} finished - Reward: {episode_reward:.2f}")
        current_episode += 1
        start_episode()
        root.after(500, run_step)
        return
    
    # Send observation to STM32
    obs_bytes = current_state.astype(np.float32).tobytes()
    ser.write(obs_bytes)
    
    # Get action back from STM32
    act_bytes = ser.read(16)
    action = np.frombuffer(act_bytes, dtype=np.float32)
    
    # Do the step
    next_state, reward, done1, done2, info = env.step(action)
    current_state = next_state
    episode_reward += reward
    current_step += 1
    
    # Show the frame
    frame = env.render()
    img = Image.fromarray(frame)
    img = img.resize((600, 400))
    photo = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo
    
    # Update status text
    status_text = f"Episode {current_episode}/3 | Step {current_step} | Reward: {episode_reward:.2f}"
    status_label.config(text=status_text)
    
    # Check if episode ended
    if done1 or done2:
        print(f"Episode {current_episode} finished - Reward: {episode_reward:.2f}")
        current_episode += 1
        start_episode()
        root.after(500, run_step)
    else:
        root.after(20, run_step)

def close_everything():
    ser.close()
    env.close()
    root.destroy()

# Start
start_episode()
root.after(100, run_step)
root.mainloop()