import tensorflow as tf
import numpy as np

# Actor Network (Policy)
class ActorNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        
        # Layer 1: state -> 128
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        
        # Layer 2: 128 -> 128
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        
        # Layer 3: 128 -> action_dim (means)
        self.mean_layer = tf.keras.layers.Dense(action_dim, activation=None)
        
        # Log standard deviation (learnable parameter)
        self.log_std = tf.Variable(tf.zeros(action_dim), trainable=True)
    
    def call(self, state):
        # Forward pass through layers
        x = self.dense1(state)  # [batch, 128]
        x = self.dense2(x)      # [batch, 128]
        mean = self.mean_layer(x)  # [batch, action_dim]
        
        return mean
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        mean = self.call(state)
        std = tf.exp(self.log_std)
        
        if deterministic:
            # Use mean directly
            action = mean
        else:
            # Sample from Gaussian: action = mean + std * noise
            noise = tf.random.normal(tf.shape(mean))
            action = mean + std * noise
        
        return action
    
    def get_action_and_logprob(self, state, action=None):
        """Get action and its log probability"""
        mean = self.call(state)
        std = tf.exp(self.log_std)
        
        if action is None:
            # Sample new action
            noise = tf.random.normal(tf.shape(mean))
            action = mean + std * noise
        
        # Calculate log probability
        # log_prob = -0.5 * [(action - mean) / std]^2 - log(std) - constant
        var = std ** 2
        log_prob = -0.5 * tf.reduce_sum(
            ((action - mean) ** 2) / var + tf.math.log(2 * np.pi * var),
            axis=-1
        )
        
        # Calculate entropy (for exploration bonus)
        entropy = 0.5 + 0.5 * tf.math.log(2 * np.pi) + tf.reduce_sum(tf.math.log(std))
        
        return action, log_prob, entropy


# Critic Network (Value function)
class CriticNetwork(tf.keras.Model):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        
        # Layer 1: state -> 128
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        
        # Layer 2: 128 -> 128
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        
        # Layer 3: 128 -> 1 (value)
        self.value_layer = tf.keras.layers.Dense(1, activation=None)
    
    def call(self, state):
        # Forward pass
        x = self.dense1(state)      # [batch, 128]
        x = self.dense2(x)          # [batch, 128]
        value = self.value_layer(x) # [batch, 1]
        
        # Return as flat array
        return tf.squeeze(value, axis=-1)  # [batch]


# Test the models
if __name__ == '__main__':
    print("Testing Actor and Critic Networks...\n")
    
    state_dim = 24  # BipedalWalker state size
    action_dim = 4  # BipedalWalker action size
    
    # Create networks
    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim)
    
    # Test with dummy data
    test_state = tf.random.normal([1, state_dim])
    
    print("Actor Network:")
    action, log_prob, entropy = actor.get_action_and_logprob(test_state)
    print(f"  Input state shape: {test_state.shape}")
    print(f"  Output action shape: {action.shape}")
    print(f"  Action values: {action.numpy()}")
    print(f"  Log probability: {log_prob.numpy()}")
    print(f"  Entropy: {entropy.numpy()}\n")
    
    print("Critic Network:")
    value = critic(test_state)
    print(f"  Input state shape: {test_state.shape}")
    print(f"  Output value: {value.numpy()}")
    print(f"  Value shape: {value.shape}\n")
    
    print("Models created successfully!")