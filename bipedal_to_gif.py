import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (avoid GPU conflicts with display)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce TensorFlow logging

import gymnasium as gym
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Ensure GPU is disabled

import numpy as np
from actor_critic_network import ActorNetwork
from PIL import Image
from datetime import datetime


def test_and_record_gif(actor, output_path='bipedal_walker.gif', num_episodes=1, 
                        fps=30, max_steps=1600, resize_width=600):
    """
    Test trained model and save animation as GIF
    
    Args:
        actor: Trained actor network
        output_path: Path to save GIF file (default: 'bipedal_walker.gif')
        num_episodes: Number of episodes to record (default: 1)
        fps: Frames per second for GIF (default: 30)
        max_steps: Maximum steps per episode (default: 1600)
        resize_width: Width to resize frames (height auto-scaled, default: 600)
        
    Returns:
        List of episode rewards
    """
    print("="*60)
    print("Recording BipedalWalker Animation")
    print("="*60)
    print(f"Episodes to record: {num_episodes}")
    print(f"Output file: {output_path}")
    print(f"GIF settings: {fps} FPS, max {max_steps} steps per episode\n")
    
    # Create environment with rgb_array mode
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    
    all_frames = []  # Store all frames from all episodes
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        episode_frames = []
        
        while not done and steps < max_steps:
            # Get action from actor network (deterministic)
            state_tensor = tf.expand_dims(state, axis=0)
            action = actor.get_action(state_tensor, deterministic=True)
            action = action.numpy()[0]
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Get frame and store it
            frame = env.render()
            episode_frames.append(frame)
            
            # Update state and statistics
            state = next_state
            total_reward += reward
            steps += 1
            
            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"   Step {steps}: Reward = {total_reward:.2f}")
        
        episode_rewards.append(total_reward)
        print(f"   Episode finished!")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Steps: {steps}")
        print(f"   Frames captured: {len(episode_frames)}")
        
        # Add episode frames to all frames
        all_frames.extend(episode_frames)
    
    env.close()
    
    # Convert frames to PIL Images and resize
    print(f"\nProcessing {len(all_frames)} frames...")
    pil_frames = []
    
    for i, frame in enumerate(all_frames):
        # Convert numpy array to PIL Image
        img = Image.fromarray(frame)
        
        # Resize to specified width (maintain aspect ratio)
        aspect_ratio = img.height / img.width
        new_height = int(resize_width * aspect_ratio)
        img = img.resize((resize_width, new_height), Image.Resampling.LANCZOS)
        
        pil_frames.append(img)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(all_frames)} frames...")
    
    # Save as GIF
    print(f"\nSaving GIF to {output_path}...")
    duration = int(1000 / fps)  # Duration per frame in milliseconds
    
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,  # 0 = infinite loop
        optimize=False  # Set to True to reduce file size (slower)
    )
    
    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n✓ GIF saved successfully!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Frames: {len(pil_frames)}")
    print(f"   Duration: {len(pil_frames) / fps:.2f} seconds")
    print(f"   FPS: {fps}")
    
    return episode_rewards


def create_optimized_gif(actor, output_path='bipedal_walker_optimized.gif', 
                        num_episodes=1, fps=20, max_steps=500, resize_width=400):
    """
    Create a smaller, optimized GIF (better for Medium articles)
    
    Tips for smaller file size:
    - Lower FPS (20 instead of 30)
    - Fewer frames (limit max_steps to 500)
    - Smaller resolution (400px instead of 600px)
    - Enable optimization
    
    Args:
        actor: Trained actor network
        output_path: Path to save GIF file
        num_episodes: Number of episodes (default: 1)
        fps: Frames per second (default: 20 for smaller size)
        max_steps: Maximum steps (default: 500 for shorter GIF)
        resize_width: Width to resize (default: 400 for smaller size)
    """
    print("Creating optimized GIF for Medium article...")
    print("(Lower FPS, shorter duration, smaller size)")
    
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    
    all_frames = []
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            state_tensor = tf.expand_dims(state, axis=0)
            action = actor.get_action(state_tensor, deterministic=True)
            action = action.numpy()[0]
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            frame = env.render()
            all_frames.append(frame)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        print(f"   Reward: {total_reward:.2f}, Steps: {steps}")
    
    env.close()
    
    # Process frames
    print(f"\nProcessing {len(all_frames)} frames...")
    pil_frames = []
    
    for frame in all_frames:
        img = Image.fromarray(frame)
        aspect_ratio = img.height / img.width
        new_height = int(resize_width * aspect_ratio)
        img = img.resize((resize_width, new_height), Image.Resampling.LANCZOS)
        pil_frames.append(img)
    
    # Save with optimization
    print(f"Saving optimized GIF...")
    duration = int(1000 / fps)
    
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=True  # Enable optimization for smaller file size
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n✓ Optimized GIF saved!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Frames: {len(pil_frames)}")
    print(f"   Duration: {len(pil_frames) / fps:.2f} seconds")
    
    return episode_rewards


def load_and_record(checkpoint_path, output_path=None, optimized=True):
    """
    Load trained model and create GIF recording
    
    Args:
        checkpoint_path: Path to saved model (without extension)
        output_path: Path for output GIF (auto-generated if None)
        optimized: If True, create smaller GIF optimized for web (default: True)
    """
    print("="*60)
    print("BipedalWalker GIF Recorder")
    print("="*60)
    print(f"Loading model from: {checkpoint_path}\n")
    
    # Create actor network
    state_dim = 24
    action_dim = 4
    
    actor = ActorNetwork(state_dim, action_dim)
    dummy_state = tf.random.normal([1, state_dim])
    _ = actor(dummy_state)
    
    # Load trained weights
    weight_path = checkpoint_path + '_actor.weights.h5'
    
    try:
        actor.load_weights(weight_path)
        print(f"✓ Weights loaded from {weight_path}\n")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        print(f"\nMake sure the file exists: {weight_path}")
        print("If you haven't trained yet, run: python train_tf_with_viewer.py")
        return
    
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if optimized:
            output_path = f'bipedal_walker_optimized_{timestamp}.gif'
        else:
            output_path = f'bipedal_walker_{timestamp}.gif'
    
    # Create GIF
    print("Recording animation...\n")
    
    try:
        if optimized:
            # Optimized version: smaller file size, good for Medium
            rewards = create_optimized_gif(
                actor, 
                output_path=output_path,
                num_episodes=1,
                fps=20,           # Lower FPS
                max_steps=500,    # Shorter duration
                resize_width=400  # Smaller size
            )
        else:
            # Full quality version: larger file, more frames
            rewards = test_and_record_gif(
                actor,
                output_path=output_path,
                num_episodes=1,
                fps=30,
                max_steps=1600,
                resize_width=600
            )
        
        print(f"\n{'='*60}")
        print("GIF Creation Complete!")
        print(f"{'='*60}")
        print(f"Average Reward: {np.mean(rewards):.2f}")
        print(f"\nYour GIF is ready for Medium: {output_path}")
        print("\nTips for Medium:")
        print("  • Drag and drop the GIF into your Medium editor")
        print("  • Medium supports GIF files up to 25MB")
        print("  • For best results, use the optimized version")
        
    except KeyboardInterrupt:
        print("\n\n✗ Recording stopped by user")
    except Exception as e:
        print(f"\n✗ Error during recording: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    # Default settings
    checkpoint_path = './save/model'
    output_path = None
    optimized = True
    
    # Command line usage:
    # python test_ppo_record_gif.py                          -> Creates optimized GIF
    # python test_ppo_record_gif.py ./save/model             -> Use specific model
    # python test_ppo_record_gif.py ./save/model my_robot.gif -> Custom output name
    # python test_ppo_record_gif.py ./save/model output.gif full -> Full quality (larger)
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3].lower() in ['full', 'f', 'high']:
        optimized = False
        print("Creating FULL QUALITY GIF (larger file size)")
    
    # Create the GIF
    load_and_record(checkpoint_path, output_path, optimized)