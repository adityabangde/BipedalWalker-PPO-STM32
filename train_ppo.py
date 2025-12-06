import os
# Set these BEFORE importing tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (avoid GPU conflicts)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import gymnasium as gym
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Ensure GPU is disabled

import numpy as np
from actor_critic_network import ActorNetwork, CriticNetwork
from ppo_agent import PPOAgent

# Import tkinter for visualization
import tkinter as tk
from PIL import Image, ImageTk


def visualize_episode(actor, episode_num):
    """
    Visualize one episode using tkinter (no segfault!)
    
    How it works:
    1. Uses rgb_array mode (returns numpy frames - NO pygame window)
    2. Displays frames in tkinter window (our own window)
    3. Updates in real-time so you can watch the robot
    
    Args:
        actor: Trained actor network
        episode_num: Current episode number (for display)
    
    Example:
        If episode_num=200 and robot walks 500 steps getting reward 150.5,
        you'll see it walking in the window with stats updating live
    """
    # Create environment with rgb_array mode (NO SEGFAULT!)
    # This returns frames as numpy arrays instead of opening pygame window
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    
    # Reset environment to start fresh
    state, _ = env.reset()
    
    # Statistics tracking
    total_reward = 0
    steps = 0
    done = False
    
    # Create tkinter window
    root = tk.Tk()
    root.title(f"Training Progress - Episode {episode_num}")
    
    # Canvas to display the robot animation
    # Size: 600x400 pixels
    canvas = tk.Canvas(root, width=600, height=400, bg='black')
    canvas.pack()
    
    # Label to show statistics
    # Example: "Episode 200 | Step 150 | Reward: 45.67"
    stats_var = tk.StringVar(value="Starting...")
    stats_label = tk.Label(root, textvariable=stats_var, font=("Arial", 12, "bold"))
    stats_label.pack(pady=5)
    
    def update_frame():
        """Update animation frame (called repeatedly)"""
        nonlocal state, total_reward, steps, done
        
        if done or steps >= 1600:
            # Episode finished
            stats_var.set(f"Episode {episode_num} | Steps: {steps} | Reward: {total_reward:.2f} | DONE")
            env.close()
            root.after(2000, root.destroy)  # Close window after 2 seconds
            return
        
        # Get action from actor network
        # Example: state=[0.5, -0.3, 0.1, ...] -> action=[0.8, -0.2, 0.5, 0.1]
        state_tensor = tf.expand_dims(state, axis=0)  # Add batch dimension: [24] -> [1, 24]
        action = actor.get_action(state_tensor, deterministic=True)  # Use mean (no randomness)
        action = action.numpy()[0]  # Remove batch dimension: [1, 4] -> [4]
        
        # Take action in environment
        # Environment returns: next_state, reward, terminated, truncated, info
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update statistics
        total_reward += reward
        steps += 1
        
        # Get frame as numpy array (rgb_array mode - no segfault!)
        # Frame shape: [400, 600, 3] - height, width, RGB channels
        frame = env.render()
        
        # Convert numpy array to PIL Image, then to tkinter PhotoImage
        img = Image.fromarray(frame)  # numpy -> PIL
        img = img.resize((600, 400), Image.Resampling.LANCZOS)  # Resize for display
        photo = ImageTk.PhotoImage(img)  # PIL -> tkinter
        
        # Display frame on canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # Keep reference (prevents garbage collection)
        
        # Update statistics text
        # Example: "Episode 200 | Step 150 | Reward: 45.67"
        stats_var.set(f"Episode {episode_num} | Step {steps} | Reward: {total_reward:.2f}")
        
        # Schedule next frame update (creates smooth animation)
        root.after(20, update_frame)  # 20ms delay = ~50 FPS
    
    # Start animation
    root.after(100, update_frame)  # Start after 100ms
    
    # Run tkinter event loop (keeps window open and responsive)
    root.mainloop()
    
    return total_reward, steps


def collect_episode(env, actor, critic, max_steps=2048):
    """
    Collect one episode of data for training
    
    Process:
    1. Reset environment
    2. For each step:
       - Get action from actor (with exploration noise)
       - Get value estimate from critic
       - Take action, observe reward
       - Store all data
    3. Return collected data for training
    
    Args:
        env: Gymnasium environment
        actor: Actor network (policy)
        critic: Critic network (value function)
        max_steps: Maximum steps per episode
    
    Returns:
        states, actions, rewards, values, log_probs, episode_reward, episode_length
        
    Example:
        Episode collects 500 steps:
        - states: [500, 24] array of robot states
        - actions: [500, 4] array of joint actions
        - rewards: [500] array of rewards (e.g., [0.3, 0.5, 0.4, ...])
        - total reward: sum = 150.5
    """
    # Storage lists for episode data
    states = []      # Robot states (position, velocity, angles, etc.)
    actions = []     # Actions taken (joint commands)
    rewards = []     # Rewards received
    values = []      # Value estimates from critic
    log_probs = []   # Log probabilities of actions (for PPO training)
    
    # Reset environment to initial state
    state, info = env.reset()
    
    episode_reward = 0  # Sum of all rewards in episode
    episode_length = 0  # Number of steps taken
    
    # Collect data step by step
    for step in range(max_steps):
        # Convert state to tensor with batch dimension
        # Example: [24] -> [1, 24]
        state_tensor = tf.expand_dims(state, axis=0)
        
        # Get action from actor with exploration noise
        # Example: mean=[0.5, -0.3, 0.2, 0.1], noise added -> action=[0.6, -0.25, 0.3, 0.05]
        action, log_prob, _ = actor.get_action_and_logprob(state_tensor)
        action_np = action.numpy()[0]       # [1, 4] -> [4]
        log_prob_np = log_prob.numpy()[0]   # [1] -> scalar
        
        # Get value estimate from critic
        # Example: critic estimates this state is worth 50.5 reward
        value = critic(state_tensor)
        value_np = value.numpy()[0]  # [1] -> scalar
        
        # Store data for training
        states.append(state)
        actions.append(action_np)
        log_probs.append(log_prob_np)
        values.append(value_np)
        
        # Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated  # Episode ends if terminated OR truncated
        
        # Store reward
        rewards.append(reward)
        episode_reward += reward
        episode_length += 1
        
        # Move to next state
        state = next_state
        
        # Stop if episode is done
        if done:
            break
    
    # Convert lists to numpy arrays for efficient processing
    states = np.array(states, dtype=np.float32)      # [length, 24]
    actions = np.array(actions, dtype=np.float32)    # [length, 4]
    rewards = np.array(rewards, dtype=np.float32)    # [length]
    values = np.array(values, dtype=np.float32)      # [length]
    log_probs = np.array(log_probs, dtype=np.float32)  # [length]
    
    return states, actions, rewards, values, log_probs, episode_reward, episode_length


def train(max_episodes=5000, max_steps_per_episode=2048, save_interval=200):
    """
    Train PPO agent on BipedalWalker
    
    Training loop:
    1. Collect episode data (states, actions, rewards)
    2. Train actor and critic networks using PPO algorithm
    3. Every save_interval episodes:
       - Save model weights
       - Visualize progress using tkinter (see robot walking!)
    4. Repeat until max_episodes reached
    
    Args:
        max_episodes: Total episodes to train (e.g., 5000)
        max_steps_per_episode: Max steps per episode (e.g., 2048)
        save_interval: Save and visualize every N episodes (e.g., 200)
        
    Example progress:
        Episode 0-199: Train and save stats
        Episode 200: Save model, show robot walking in tkinter window
        Episode 201-399: Train and save stats
        Episode 400: Save model, show robot walking again
        ...and so on
    """
    print("="*60)
    print("PPO Training on BipedalWalker-v3")
    print("="*60)
    print(f"Max Episodes: {max_episodes}")
    print(f"Max Steps per Episode: {max_steps_per_episode}")
    print(f"Save & Visualize Interval: {save_interval}")
    print("="*60 + "\n")
    
    # Create environment
    env = gym.make('BipedalWalker-v3')
    state_dim = env.observation_space.shape[0]  # 24 (robot state size)
    action_dim = env.action_space.shape[0]      # 4 (joint actions)
    
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}\n")
    
    # Create actor and critic networks
    print("Creating Actor and Critic networks...")
    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim)
    
    # Build networks by calling them once (initializes layers)
    dummy_state = tf.random.normal([1, state_dim])
    _ = actor(dummy_state)
    _ = critic(dummy_state)
    print("Networks created!\n")
    
    # Create PPO agent (handles training logic)
    print("Creating PPO agent...")
    agent = PPOAgent(
        actor=actor,
        critic=critic,
        lr=3e-4,           # Learning rate (how fast to update weights)
        clip_ratio=0.2,    # PPO clipping (prevents too large updates)
        epochs=4,          # Training epochs per update (iterate over data 4 times)
        batch_size=64,     # Mini-batch size (process 64 samples at a time)
        entropy_coef=0.01  # Entropy bonus (encourages exploration)
    )
    print("Agent created!\n")
    
    # Create save directory for model checkpoints
    save_dir = './save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created save directory: {save_dir}\n")
    
    # Training statistics (running averages)
    running_reward = 0   # Sum of rewards over save_interval episodes
    running_length = 0   # Sum of lengths over save_interval episodes
    
    print("Starting training...\n")
    print("="*60)
    
    # Main training loop
    for episode in range(max_episodes):
        # Step 1: Collect one episode of data
        # Example: Robot runs for 500 steps, collects states, actions, rewards
        states, actions, rewards, values, log_probs, ep_reward, ep_length = \
            collect_episode(env, actor, critic, max_steps=max_steps_per_episode)
        
        # Step 2: Train agent with collected data
        # PPO updates actor and critic to improve policy
        # Example: actor_loss=0.05 means policy is improving, critic_loss=2.5 means value estimates improving
        actor_loss, critic_loss, entropy = agent.train(
            states, actions, rewards, values, log_probs, gamma=0.99
        )
        
        # Step 3: Update running statistics
        running_reward += ep_reward  # Accumulate reward
        running_length += ep_length  # Accumulate length
        
        # Step 4: Print progress for this episode
        # Example: "Episode 150 | Reward: 123.45 | Length: 500 | Actor Loss: 0.0234 | ..."
        print(f"Episode {episode:4d} | "
              f"Reward: {ep_reward:7.2f} | "
              f"Length: {ep_length:4d} | "
              f"Actor Loss: {actor_loss:7.4f} | "
              f"Critic Loss: {critic_loss:7.4f} | "
              f"Entropy: {entropy:6.4f}")
        
        # Step 5: Save and visualize every save_interval episodes
        if (episode + 1) % save_interval == 0:
            print("\n" + "-"*60)
            print(f"Checkpoint at Episode {episode + 1}/{max_episodes}")
            print("-"*60)
            
            # Calculate average statistics over last save_interval episodes
            # Example: If save_interval=200, avg_reward is average over episodes 0-199
            avg_reward = running_reward / save_interval
            avg_length = running_length / save_interval
            
            print(f"Average Reward (last {save_interval} episodes): {avg_reward:.2f}")
            print(f"Average Length (last {save_interval} episodes): {avg_length:.2f}")
            
            # Save model weights
            print(f"\nSaving model...")
            checkpoint_path = os.path.join(save_dir, 'model')
            agent.save(checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
            
            # Visualize current policy using tkinter (NO SEGFAULT!)
            # Opens window showing robot walking with current trained policy
            print(f"\nVisualizing episode {episode + 1}...")
            print("(Tkinter window will open - close it when done watching)")
            
            try:
                vis_reward, vis_steps = visualize_episode(actor, episode + 1)
                print(f"Visualization: Reward={vis_reward:.2f}, Steps={vis_steps}")
            except Exception as e:
                print(f"Visualization failed: {e}")
                print("(Training continues normally)")
            
            print("-"*60 + "\n")
            
            # Reset running statistics for next interval
            running_reward = 0
            running_length = 0
    
    # Training complete!
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # Final save
    print(f"\nSaving final model...")
    checkpoint_path = os.path.join(save_dir, 'model_final')
    agent.save(checkpoint_path)
    print(f"Final model saved to {checkpoint_path}")
    
    # Final visualization
    print(f"\nFinal visualization...")
    try:
        vis_reward, vis_steps = visualize_episode(actor, max_episodes)
        print(f"Final performance: Reward={vis_reward:.2f}, Steps={vis_steps}")
    except Exception as e:
        print(f"Final visualization failed: {e}")
    
    env.close()
    print("\nTraining finished successfully!")
    print("Run 'python test_tkinter_simple.py' to test your trained model anytime!")


if __name__ == '__main__':
    # Train the agent
    # Every 200 episodes: saves model + shows robot walking in tkinter window
    train(
        max_episodes=5000,           # Total episodes to train
        max_steps_per_episode=2048,  # Max steps per episode
        save_interval=200            # Save & visualize every 200 episodes
    )