import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np
import gymnasium as gym
import tkinter as tk
from PIL import Image, ImageTk
from actor_critic_network import ActorNetwork

def test_tflite_model(tflite_model_path, num_episodes=3, visualize=True):
    """Test TFLite model with optional visualization"""
    
    print(f"\nTesting: {tflite_model_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    is_quantized = input_details[0]['dtype'] == np.int8
    
    if is_quantized:
        input_scale = input_details[0]['quantization'][0]
        input_zero_point = input_details[0]['quantization'][1]
        output_scale = output_details[0]['quantization'][0]
        output_zero_point = output_details[0]['quantization'][1]
        print(f"Model type: Quantized INT8")
    else:
        print(f"Model type: Float32")
    
    # Inference function
    def run_inference(state):
        state_input = state.reshape(1, 24).astype(np.float32)
        
        if is_quantized:
            state_input = (state_input / input_scale + input_zero_point).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], state_input)
        interpreter.invoke()
        action_output = interpreter.get_tensor(output_details[0]['index'])
        
        if is_quantized:
            action = (action_output.astype(np.float32) - output_zero_point) * output_scale
        else:
            action = action_output
        
        return action[0]
    
    # Test with or without visualization
    if visualize:
        avg_reward = test_with_visualization(run_inference, tflite_model_path, num_episodes)
    else:
        env = gym.make('BipedalWalker-v3')
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1600:
                action = run_inference(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        env.close()
        avg_reward = np.mean(episode_rewards)
        print(f"Average Reward: {avg_reward:.2f} ± {np.std(episode_rewards):.2f}")
    
    return avg_reward


def test_with_visualization(inference_fn, model_name, num_episodes=3):
    """Test TFLite model with tkinter visualization"""
    
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    
    root = tk.Tk()
    model_type = "Quantized INT8" if "quantized" in model_name else "Float32"
    root.title(f"TFLite Test - {model_type}")
    
    canvas = tk.Canvas(root, width=600, height=400, bg='black')
    canvas.pack()
    
    stats_var = tk.StringVar(value="Starting...")
    stats_label = tk.Label(root, textvariable=stats_var, font=("Arial", 14, "bold"))
    stats_label.pack(pady=10)
    
    # Animation state
    current_episode = [0]
    state = [None]
    total_reward = [0.0]
    steps = [0]
    done = [False]
    episode_rewards = []
    
    def start_new_episode():
        if current_episode[0] >= num_episodes:
            avg_reward = np.mean(episode_rewards)
            stats_var.set(f"Complete! Avg Reward: {avg_reward:.2f}")
            print(f"Average Reward: {avg_reward:.2f}")
            root.after(2000, root.destroy)
            return
        
        current_episode[0] += 1
        state[0], _ = env.reset()
        total_reward[0] = 0.0
        steps[0] = 0
        done[0] = False
        
        print(f"Episode {current_episode[0]}/{num_episodes}")
        root.after(10, update_frame)
    
    def update_frame():
        if done[0]:
            episode_rewards.append(total_reward[0])
            print(f"Reward: {total_reward[0]:.2f}, Steps: {steps[0]}")
            root.after(500, start_new_episode)
            return
        
        if steps[0] >= 1600:
            done[0] = True
            root.after(10, update_frame)
            return
        
        # Get action and step
        action = inference_fn(state[0])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done[0] = terminated or truncated
        
        # Update state
        state[0] = next_state
        total_reward[0] += reward
        steps[0] += 1
        
        # Render frame
        rgb_frame = env.render()
        img = Image.fromarray(rgb_frame)
        img = img.resize((600, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.photo = photo
        
        # Update stats
        stats_var.set(f"Episode {current_episode[0]}/{num_episodes} | Step {steps[0]} | Reward: {total_reward[0]:.2f}")
        
        root.after(10, update_frame)
    
    start_new_episode()
    root.mainloop()
    env.close()
    
    return np.mean(episode_rewards)


def test_original_model(checkpoint_path, num_episodes=3, visualize=True):
    """Test original TensorFlow model"""
    
    print(f"\nTesting original model: {checkpoint_path}")
    
    state_dim = 24
    action_dim = 4
    actor = ActorNetwork(state_dim, action_dim)
    
    dummy_state = tf.random.normal([1, state_dim])
    _ = actor(dummy_state)
    
    actor_weights_path = checkpoint_path + '_actor.weights.h5'
    actor.load_weights(actor_weights_path)
    
    def run_inference(state):
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        action = actor(state_tensor)
        return action.numpy()[0]
    
    if visualize:
        avg_reward = test_with_visualization(run_inference, "original_tf", num_episodes)
    else:
        env = gym.make('BipedalWalker-v3')
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1600:
                action = run_inference(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        env.close()
        avg_reward = np.mean(episode_rewards)
        print(f"Average Reward: {avg_reward:.2f} ± {np.std(episode_rewards):.2f}")
    
    return avg_reward


def compare_models(checkpoint_path, tflite_dir, num_episodes=3, visualize=True):
    """Compare original, float32, and quantized models"""
    
    results = {}
    
    # Test original
    print("\n" + "="*60)
    print("TEST 1: ORIGINAL TENSORFLOW MODEL")
    print("="*60)
    
    actor_weights_path = checkpoint_path + '_actor.weights.h5'
    if os.path.exists(actor_weights_path):
        try:
            results['original'] = test_original_model(checkpoint_path, num_episodes, visualize)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Model not found: {actor_weights_path}")
    
    # Test Float32
    print("\n" + "="*60)
    print("TEST 2: FLOAT32 TFLITE MODEL")
    print("="*60)
    
    float32_path = os.path.join(tflite_dir, 'actor_model_float32.tflite')
    if os.path.exists(float32_path):
        try:
            results['float32'] = test_tflite_model(float32_path, num_episodes, visualize)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Model not found: {float32_path}")
    
    # Test Quantized
    print("\n" + "="*60)
    print("TEST 3: QUANTIZED INT8 TFLITE MODEL")
    print("="*60)
    
    quantized_path = os.path.join(tflite_dir, 'actor_model_quantized.tflite')
    if os.path.exists(quantized_path):
        try:
            results['quantized'] = test_tflite_model(quantized_path, num_episodes, visualize)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Model not found: {quantized_path}")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if results:
        for name, reward in results.items():
            print(f"{name:15s}: {reward:7.2f}")
        
        if 'original' in results and 'quantized' in results:
            diff = results['quantized'] - results['original']
            pct = (diff / results['original']) * 100
            print(f"\nQuantized vs Original: {diff:+.2f} ({pct:+.1f}%)")


def quick_inference_test(tflite_model_path):
    """Quick single inference test"""
    
    print(f"\nQuick test: {tflite_model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    is_quantized = input_details[0]['dtype'] == np.int8
    
    test_state = np.random.uniform(-5.0, 5.0, size=(1, 24)).astype(np.float32)
    
    if is_quantized:
        input_scale = input_details[0]['quantization'][0]
        input_zero_point = input_details[0]['quantization'][1]
        test_state = (test_state / input_scale + input_zero_point).astype(np.int8)
    
    import time
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], test_state)
    interpreter.invoke()
    action = interpreter.get_tensor(output_details[0]['index'])
    inference_time = (time.time() - start) * 1000
    
    if is_quantized:
        output_scale = output_details[0]['quantization'][0]
        output_zero_point = output_details[0]['quantization'][1]
        action = (action.astype(np.float32) - output_zero_point) * output_scale
    
    print(f"Inference time: {inference_time:.3f} ms")
    print(f"Output action: {action[0]}")


if __name__ == '__main__':
    import sys
    
    checkpoint_path = './save/model'
    tflite_dir = './tflite_models'
    num_episodes = 3
    visualize = True
    
    # Command line arguments
    if len(sys.argv) > 1 and sys.argv[1] != '--no-viz':
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] != '--no-viz':
        num_episodes = int(sys.argv[2])
    if '--no-viz' in sys.argv:
        visualize = False
    
    print(f"\nConfiguration:")
    print(f"  Original: {checkpoint_path}_actor.weights.h5")
    print(f"  TFLite: {tflite_dir}/")
    print(f"  Episodes: {num_episodes}")
    print(f"  Visualization: {visualize}")
    
    try:
        # Quick tests
        print("\n" + "="*60)
        print("QUICK INFERENCE TESTS")
        print("="*60)
        
        float32_path = os.path.join(tflite_dir, 'actor_model_float32.tflite')
        quantized_path = os.path.join(tflite_dir, 'actor_model_quantized.tflite')
        
        if os.path.exists(float32_path):
            quick_inference_test(float32_path)
        if os.path.exists(quantized_path):
            quick_inference_test(quantized_path)
        
        # Full comparison
        print("\n" + "="*60)
        print("FULL EPISODE COMPARISON")
        print("="*60)
        
        compare_models(checkpoint_path, tflite_dir, num_episodes, visualize)
        
        print("\nTesting complete!")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()