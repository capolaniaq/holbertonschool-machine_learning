#!/usr/bin/env python3
"""
Train model
"""
import gym
import numpy as np
import tensorflow.keras as K
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor


num_actions = 4
def create_q_model():
    """
    Create model with convolutional networks
    """
    inputs = layers.Input(shape=(84, 84, 4,))

    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

class AtariProcessor(Processor):
    """
    Defines Atari environment to play Breakout
    """
    def process_observation(self, observation):
        """
        Resizes images and makes grayscale to conserve memory
        """
        assert observation.ndim == 3
        image = Image.fromarray(observation)
        image = image.resize((84, 84), Image.ANTIALIAS).convert('L')
        processed_observation = np.array(image)

        assert processed_observation.shape == (84, 84)

        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """
        Converts a batch of images to float32
        """
        processed_batch = batch.astype('float32') / 255.0
        return processed_batch

    def process_reward(self, reward):
        """
        Processes reward between -1 and 1
        """
        return np.clip(reward, -1., 1.)


def training():
    """
    Train the model for play breackout
    """
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []

    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0

    while True:
        state = np.array(env.reset())
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):

            frame_count += 1

            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                action = np.random.choice(num_actions)
            else:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                action = tf.argmax(action_probs[0]).numpy()

            if frame_count < epsilon_greedy_frames:
            epsilon -= epsilon_interval_1 / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min_1)
            
            if frame_count > epsilon_greedy_frames and frame_count < 2 * epsilon_greedy_frames:
            epsilon -= epsilon_interval_2 / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min_2)
            
            if frame_count > 2 * epsilon_greedy_frames:
            epsilon -= epsilon_interval_3 / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min_3)

            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                future_rewards = model_target.predict(state_next_sample)
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    q_values = model(state_sample)
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    loss = loss_function(updated_q_values, q_action)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}, epsilon {:.3f}, loss {:.5f}"
                print(template.format(running_reward, episode_count, frame_count, epsilon, loss))

            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

        if running_reward > 18:
            print("Solved at episode {}!".format(episode_count))
            break

if __name__ == '__main__':
    training()
