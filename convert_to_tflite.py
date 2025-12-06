import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np
from actor_critic_network import ActorNetwork

def convert_actor_to_tflite(checkpoint_path, output_dir='./tflite_models', quantize=True):
    """Convert trained Actor network to TFLite format for MCU deployment"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load actor network
    state_dim = 24
    action_dim = 4
    actor = ActorNetwork(state_dim, action_dim)
    
    dummy_state = tf.random.normal([1, state_dim])
    _ = actor(dummy_state)
    
    actor_weights_path = checkpoint_path + '_actor.weights.h5'
    actor.load_weights(actor_weights_path)
    print(f"Loaded weights from: {actor_weights_path}")
    
    # Create concrete function
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 24], dtype=tf.float32)])
    def actor_inference(state):
        return actor.call(state)
    
    concrete_func = actor_inference.get_concrete_function()
    
    # Convert to Float32
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = []
    tflite_model_float32 = converter.convert()
    
    float32_path = os.path.join(output_dir, 'actor_model_float32.tflite')
    with open(float32_path, 'wb') as f:
        f.write(tflite_model_float32)
    
    float32_size = len(tflite_model_float32) / 1024
    print(f"Float32 model: {float32_size:.2f} KB -> {float32_path}")
    
    # Convert to Quantized INT8
    quantized_path = None
    if quantize:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_dataset():
            for _ in range(100):
                state = np.random.uniform(-5.0, 5.0, size=(1, 24)).astype(np.float32)
                yield [state]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        try:
            tflite_model_quantized = converter.convert()
            quantized_path = os.path.join(output_dir, 'actor_model_quantized.tflite')
            
            with open(quantized_path, 'wb') as f:
                f.write(tflite_model_quantized)
            
            quantized_size = len(tflite_model_quantized) / 1024
            print(f"Quantized model: {quantized_size:.2f} KB -> {quantized_path}")
            print(f"Size reduction: {float32_size/quantized_size:.1f}x")
        except Exception as e:
            print(f"Quantization failed: {e}")
    
    return float32_path, quantized_path


if __name__ == '__main__':
    import sys
    
    checkpoint_path = './save/model'
    output_dir = './tflite_models'
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print(f"\nConverting: {checkpoint_path}_actor.weights.h5")
    print(f"Output: {output_dir}/\n")
    
    try:
        float32_path, quantized_path = convert_actor_to_tflite(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            quantize=True
        )
        
        if float32_path:
            print("\nConversion successful!")
            print("Run 'python test_tflite_model.py' to test the models")
        else:
            print("Conversion failed!")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()