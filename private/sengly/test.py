# Import required libraries
import tensorflow as tf
import numpy as np

# Constants
GRID_SIZE = 5
NUM_TREASURES = 3
num_actions = 4

# Create the Q-network
def create_q_network():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(GRID_SIZE, GRID_SIZE, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', trainable=True),
        tf.keras.layers.Dense(num_actions, trainable=True)
    ])
    return model

# Initialize the Q-network and optimizer
q_network = create_q_network()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Function to train a single episode
@tf.function
def train_episode(q_network, optimizer, state, epsilon):
    treasure_locations = np.array([[2, 2], [1, 4], [4, 3]])  # Define the treasure locations
    next_state = tf.identity(state)  # Initialize next_state

    with tf.GradientTape() as tape:
        state_tensor = tf.cast(state, tf.float32)
        state_tensor = tf.repeat(state_tensor, GRID_SIZE, axis=0)
        state_tensor = tf.repeat(state_tensor, GRID_SIZE, axis=1)
        state_tensor = tf.expand_dims(state_tensor, axis=0)

        # Get action values from the Q-network
        action_values = q_network(state_tensor)
        action = tf.argmax(action_values[0])  # Choose the action with the highest value

        # Explore randomly with a probability of epsilon
        if np.random.random() < epsilon:
            action = np.random.randint(num_actions)

        # Update next_state based on the chosen action
        if action == 0:  # Move left
            next_state = tf.tensor_scatter_nd_update(next_state, [[0, 0, 1]], [tf.subtract(next_state[0, 0, 1], 1)])
        elif action == 1:  # Move right
            next_state = tf.tensor_scatter_nd_update(next_state, [[0, 0, 1]], [next_state[0, 0, 1] + 1])
        elif action == 2:  # Move up
            next_state = tf.tensor_scatter_nd_update(next_state, [[0, 0, 0]], [tf.subtract(next_state[0, 0, 0], 1)])
        elif action == 3:  # Move down
            next_state = tf.tensor_scatter_nd_update(next_state, [[0, 0, 0]], [next_state[0, 0, 0] + 1])

        # Calculate the reward based on whether the agent reached a treasure location
        reward = 0.0
        for i in range(NUM_TREASURES):
            if np.all(next_state[0, 0, 0:2] == treasure_locations[i]):
                reward = 1.0
                break

        # Clip the coordinates of next_state to ensure they stay within the grid boundaries
        next_state = tf.tensor_scatter_nd_update(next_state, [[0, 0, 0]], [tf.clip_by_value(next_state[0, 0, 0], 0, GRID_SIZE - 1)])
        next_state = tf.tensor_scatter_nd_update(next_state, [[0, 0, 1]], [tf.clip_by_value(next_state[0, 0, 1], 0, GRID-1)])

        # Preprocess the next state for input to the Q-network
        next_state_tensor = tf.cast(next_state, tf.float32)
        next_state_tensor = tf.repeat(next_state_tensor, GRID_SIZE, axis=0)
        next_state_tensor = tf.repeat(next_state_tensor, GRID_SIZE, axis=1)
        next_state_tensor = tf.expand_dims(next_state_tensor, axis=0)

        # Calculate the target value for the Q-network update
        target = reward + 0.99 * tf.reduce_max(q_network(next_state_tensor), axis=1)

        # Create a one-hot mask for the chosen action
        mask = tf.one_hot(action, num_actions)
        action_values_masked = tf.reduce_sum(action_values * mask, axis=1)

        # Calculate the loss between the predicted and target Q-values
        loss = tf.keras.losses.MeanSquaredError()(target, action_values_masked)

    # Calculate gradients and update the Q-network weights
    grads = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

    # Update the current state to the next state
    state = next_state

    return reward

# Initialize the state and exploration factor
state = np.array([[[2, 2], [2, 2]]], dtype=np.int64)
epsilon = 1.0

# Training loop
NUM_EPISODES = 100
for episode in range(NUM_EPISODES):
    total_reward = train_episode(q_network, optimizer, state, epsilon)

    # Decay the exploration factor after each episode
    epsilon *= 0.99

    print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")
