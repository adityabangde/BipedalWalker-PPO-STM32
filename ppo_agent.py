import tensorflow as tf
import numpy as np

class PPOAgent:
    def __init__(self, actor, critic, lr=3e-4, clip_ratio=0.2, 
                 epochs=4, batch_size=64, entropy_coef=0.01):
        """
        PPO Agent
        
        actor: Actor network
        critic: Critic network
        lr: Learning rate
        clip_ratio: PPO clipping parameter (0.2 means clip between 0.8 to 1.2)
        epochs: Number of training epochs per update
        batch_size: Mini-batch size
        entropy_coef: Entropy bonus coefficient
        """
        self.actor = actor
        self.critic = critic
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    def calculate_returns(self, rewards, gamma=0.99):
        """
        Calculate discounted returns
        Return[t] = Reward[t] + gamma * Return[t+1]
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        
        # Go backwards from end
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def calculate_advantages(self, returns, values):
        """
        Calculate advantages
        Advantage = Return - Value_predicted
        Then normalize: (adv - mean) / (std + 1e-8)
        """
        advantages = returns - values
        
        # Normalize
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages
    
    @tf.function
    def train_actor_step(self, states, actions, old_log_probs, advantages):
        """
        Train actor network for one step
        """
        with tf.GradientTape() as tape:
            # Get new log probs with current policy
            _, new_log_probs, entropy = self.actor.get_action_and_logprob(states, actions)
            
            # Calculate ratio = exp(new_log_prob - old_log_prob)
            ratio = tf.exp(new_log_probs - old_log_probs)
            
            # PPO clipped objective
            # loss1 = -advantage * ratio
            # loss2 = -advantage * clip(ratio, 0.8, 1.2)
            # loss = max(loss1, loss2)
            
            clipped_ratio = tf.clip_by_value(ratio, 
                                             1.0 - self.clip_ratio, 
                                             1.0 + self.clip_ratio)
            
            loss1 = -advantages * ratio
            loss2 = -advantages * clipped_ratio
            policy_loss = tf.reduce_mean(tf.maximum(loss1, loss2))
            
            # Add entropy bonus (encourages exploration)
            entropy_loss = -self.entropy_coef * entropy
            
            # Total actor loss
            actor_loss = policy_loss + entropy_loss
        
        # Backpropagation
        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
        
        return actor_loss, policy_loss, entropy
    
    @tf.function
    def train_critic_step(self, states, returns, old_values):
        """
        Train critic network for one step
        """
        with tf.GradientTape() as tape:
            # Get new value predictions
            new_values = self.critic(states)
            
            # PPO clipped value loss
            # Clip the value change
            clipped_values = old_values + tf.clip_by_value(
                new_values - old_values,
                -self.clip_ratio,
                self.clip_ratio
            )
            
            # Calculate loss
            # loss1 = (return - new_value)^2
            # loss2 = (return - clipped_value)^2
            # loss = max(loss1, loss2)
            
            loss1 = (returns - new_values) ** 2
            loss2 = (returns - clipped_values) ** 2
            critic_loss = tf.reduce_mean(tf.maximum(loss1, loss2))
        
        # Backpropagation
        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        
        return critic_loss
    
    def train(self, states, actions, rewards, values, old_log_probs, gamma=0.99):
        """
        Train the agent with collected data
        
        Steps:
        1. Calculate returns (targets for critic)
        2. Calculate advantages
        3. Train actor and critic for multiple epochs with mini-batches
        """
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        
        # Step 1: Calculate returns
        returns = self.calculate_returns(rewards, gamma)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Step 2: Calculate advantages
        advantages = self.calculate_advantages(returns.numpy(), values)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
        # Old values as tensor
        old_values = tf.convert_to_tensor(values, dtype=tf.float32)
        
        # Step 3: Train for multiple epochs
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        actor_losses = []
        critic_losses = []
        entropies = []
        
        for epoch in range(self.epochs):
            # Shuffle data
            np.random.shuffle(indices)
            
            # Train on mini-batches
            num_batches = max(1, dataset_size // self.batch_size)
            
            for i in range(num_batches):
                # Get batch indices
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = tf.gather(states, batch_indices)
                batch_actions = tf.gather(actions, batch_indices)
                batch_old_log_probs = tf.gather(old_log_probs, batch_indices)
                batch_advantages = tf.gather(advantages, batch_indices)
                batch_returns = tf.gather(returns, batch_indices)
                batch_old_values = tf.gather(old_values, batch_indices)
                
                # Train actor
                actor_loss, policy_loss, entropy = self.train_actor_step(
                    batch_states, batch_actions, batch_old_log_probs, batch_advantages
                )
                
                # Train critic
                critic_loss = self.train_critic_step(
                    batch_states, batch_returns, batch_old_values
                )
                
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
                entropies.append(entropy.numpy())
        
        # Return average losses
        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropies)
    
    def save(self, filepath):
        """Save actor and critic weights"""
        self.actor.save_weights(filepath + '_actor.weights.h5')
        self.critic.save_weights(filepath + '_critic.weights.h5')
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load actor and critic weights"""
        self.actor.load_weights(filepath + '_actor.weights.h5')
        self.critic.load_weights(filepath + '_critic.weights.h5')
        print(f"Model loaded from {filepath}")