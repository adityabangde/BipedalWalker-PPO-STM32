import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (avoid GPU conflicts with display)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce TensorFlow logging

import gymnasium as gym
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Ensure GPU is disabled

import numpy as np
from actor_critic_network import ActorNetwork
import tkinter as tk
from PIL import Image, ImageTk


def test_with_tkinter(actor, num_episodes=3):
    """
    Test trained model using tkinter viewer (no segfault!)
    
    How it works:
    1. Creates environment with rgb_array mode (returns frames, no pygame window)
    2. For each episode:
       - Runs the trained policy (actor network)
       - Gets frames as numpy arrays
       - Displays frames in tkinter window in real-time
    3. Shows statistics as robot walks
    
    Args:
        actor: Trained actor network
        num_episodes: Number of episodes to test (default 3)
        
    Example:
        If testing 3 episodes:
        - Episode 1: Opens window, robot walks, shows reward (e.g., 250.5)
        - Episode 2: Same window updates, new episode (e.g., 275.3)
        - Episode 3: Final episode (e.g., 260.8)
        - Window closes automatically when done
    """
    # Create environment with rgb_array mode (NO SEGFAULT!)
    # Returns frames as numpy arrays instead of creating pygame window
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    
    # Create tkinter window for visualization
    root = tk.Tk()
    root.title("BipedalWalker - Testing Trained Model")
    
    # Canvas to display robot animation (600x400 pixels)
    canvas = tk.Canvas(root, width=600, height=400, bg='black')
    canvas.pack()
    
    # Label to show statistics
    # Example: "Episode 1/3 | Step 150 | Reward: 45.67"
    stats_var = tk.StringVar(value="Starting tests...")
    stats_label = tk.Label(root, textvariable=stats_var, font=("Arial", 14, "bold"))
    stats_label.pack(pady=10)
    
    # Variables for animation loop
    current_episode = [0]        # Current episode number (list for mutability in nested function)
    state = [None]               # Current state
    total_reward = [0.0]         # Total reward for current episode
    steps = [0]                  # Steps in current episode
    done = [False]               # Whether episode is done
    episode_rewards = []         # List to store all episode rewards
    
    def start_new_episode():
        """Start a new episode"""
        if current_episode[0] >= num_episodes:
            # All episodes complete
            avg_reward = np.mean(episode_rewards)
            stats_var.set(f"Complete! Tested {num_episodes} episodes | Avg Reward: {avg_reward:.2f}")
            print(f"\nTesting complete!")
            print(f"   Episodes: {num_episodes}")
            print(f"   Average Reward: {avg_reward:.2f}")
            print(f"   Episode Rewards: {[f'{r:.2f}' for r in episode_rewards]}")
            root.after(3000, root.destroy)  # Close window after 3 seconds
            return
        
        # Reset environment for new episode
        current_episode[0] += 1
        state[0], _ = env.reset()
        total_reward[0] = 0.0
        steps[0] = 0
        done[0] = False
        
        print(f"\nEpisode {current_episode[0]}/{num_episodes}")
        print("-" * 40)
        
        # Start animation loop for this episode
        root.after(10, update_frame)
    
    def update_frame():
        """Update animation frame (called repeatedly for smooth animation)"""
        if done[0]:
            # Episode finished
            episode_rewards.append(total_reward[0])
            print(f"   Episode {current_episode[0]} finished!")
            print(f"   Reward: {total_reward[0]:.2f}")
            print(f"   Steps: {steps[0]}")
            
            # Start next episode after short delay
            root.after(500, start_new_episode)
            return
        
        if steps[0] >= 1600:
            # Max steps reached
            done[0] = True
            root.after(10, update_frame)
            return
        
        # Get action from actor network (deterministic - no randomness)
        # Example: state=[0.5, -0.3, 0.1, ...] -> action=[0.8, -0.2, 0.5, 0.1]
        state_tensor = tf.expand_dims(state[0], axis=0)  # Add batch dimension: [24] -> [1, 24]
        action = actor.get_action(state_tensor, deterministic=True)  # Use mean (no exploration)
        action = action.numpy()[0]  # Remove batch dimension: [1, 4] -> [4]
        
        # Take action in environment
        # Returns: next_state, reward, terminated, truncated, info
        next_state, reward, terminated, truncated, _ = env.step(action)
        done[0] = terminated or truncated
        
        # Update statistics
        state[0] = next_state
        total_reward[0] += reward
        steps[0] += 1
        
        # Get frame as numpy array (rgb_array mode - no pygame window!)
        # Frame shape: [400, 600, 3] - height, width, RGB channels
        frame = env.render()
        
        # Convert numpy array to displayable image
        # numpy -> PIL Image -> tkinter PhotoImage
        img = Image.fromarray(frame)  # Create PIL Image from numpy array
        img = img.resize((600, 400), Image.Resampling.LANCZOS)  # Resize for display
        photo = ImageTk.PhotoImage(img)  # Convert to tkinter format
        
        # Display frame on canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # Keep reference (prevents garbage collection)
        
        # Update statistics text
        stats_var.set(
            f"Episode {current_episode[0]}/{num_episodes} | "
            f"Step {steps[0]} | "
            f"Reward: {total_reward[0]:.2f}"
        )
        
        # Print progress every 100 steps
        if steps[0] % 100 == 0:
            print(f"   Step {steps[0]}: Reward = {total_reward[0]:.2f}")
        
        # Schedule next frame update (creates smooth animation)
        root.after(20, update_frame)  # 20ms delay = ~50 FPS
    
    # Start first episode
    root.after(100, start_new_episode)
    
    # Run tkinter event loop (keeps window open and responsive)
    root.mainloop()
    
    env.close()


def load_and_test(checkpoint_path, num_episodes=3):
    """
    Load trained model and test it with tkinter visualization
    
    Steps:
    1. Create actor network
    2. Load saved weights
    3. Test with tkinter viewer (shows robot walking)
    
    Args:
        checkpoint_path: Path to saved model (without extension)
                        Example: './save/model' will load './save/model_actor.weights.h5'
        num_episodes: Number of episodes to test
        
    Example:
        load_and_test('./save/model', 3)
        -> Loads model from ./save/model_actor.weights.h5
        -> Opens tkinter window
        -> Shows robot walking for 3 episodes
        -> Displays rewards and statistics
    """
    print("="*60)
    print("Testing Trained BipedalWalker Model")
    print("="*60)
    print(f"Loading model from: {checkpoint_path}")
    print(f"Number of test episodes: {num_episodes}\n")
    
    # Create actor network
    state_dim = 24  # BipedalWalker state dimension
    action_dim = 4  # BipedalWalker action dimension
    
    print("Creating actor network...")
    actor = ActorNetwork(state_dim, action_dim)
    
    # Build network by calling it once (initializes layers)
    dummy_state = tf.random.normal([1, state_dim])
    _ = actor(dummy_state)
    print("Network created\n")
    
    # Load trained weights
    print("Loading trained weights...")
    weight_path = checkpoint_path + '_actor.weights.h5'
    
    try:
        actor.load_weights(weight_path)
        print(f"Weights loaded from {weight_path}\n")
    except Exception as e:
        print(f" Error loading weights: {e}")
        print(f"\n Make sure the file exists: {weight_path}")
        print("If you haven't trained yet, run: python train_tf_with_viewer.py")
        return
    
    # Test the model with tkinter visualization
    print("Starting visualization...")
    
    try:
        test_with_tkinter(actor, num_episodes)
    except KeyboardInterrupt:
        print("\n\n  Testing stopped by user")
    except Exception as e:
        print(f"\n Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    # Default settings
    checkpoint_path = './save/model'  # Path to saved model
    num_episodes = 3                   # Number of episodes to test
    
    # Allow command line arguments
    # Usage: python test_tkinter.py [checkpoint_path] [num_episodes]
    # Example: python test_tkinter.py ./save/model_final 5
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        num_episodes = int(sys.argv[2])
    
    # Load and test the model
    load_and_test(checkpoint_path, num_episodes)
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)