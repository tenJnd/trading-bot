import logging
import os
import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gym import spaces
from jnd_utils.log import init_logging

from agent_training_data import prepare_training_data
from src.exchange_adapter import BaseExchangeAdapter
from src.exchange_factory import ExchangeFactory

# Set up TensorFlow GPU memory management
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

_logger = logging.getLogger(__name__)


# Define the Trading Environment
class TradingEnv(gym.Env):
    """A trading environment for reinforcement learning with dynamic features."""

    def __init__(self, data_shape_len, initial_investment=1_000):
        super(TradingEnv, self).__init__()
        self.orig_data = ...
        self.data = ...
        self.initial_investment = initial_investment
        self.current_step = 0
        self.done = False
        self.position = 0  # -1: short, 0: no position, 1: long
        self.entry_price = 0
        self.balance = initial_investment
        self.asset_held = 0
        self.step_returns = []
        self.max_balance = initial_investment
        self.action_space = spaces.Discrete(4)  # 0: hold, 1: buy, 2: sell, 3: close
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(data_shape_len,),
                                            dtype=np.float32)

    def step(self, action):
        """
        Executes a step in the environment based on the agent's action.

        Args:
            action (int): The action taken by the agent.
                          0: Hold, 1: Buy/Long, 2: Sell/Short

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Save the previous balance to calculate returns and rewards
        previous_balance = self.balance

        # Check if we've reached the end of the dataset
        if self.current_step >= len(self.data) - 1:
            self.done = True

        # Get the current market price
        current_price = self.orig_data.iloc[self.current_step]['C']
        # Fetch the previous candle's high and low
        previous_low = self.orig_data.iloc[self.current_step - 1]['L'] if self.current_step > 0 else self.entry_price
        previous_high = self.orig_data.iloc[self.current_step - 1]['H'] if self.current_step > 0 else self.entry_price
        atr = self.orig_data.iloc[self.current_step]['atr_20']  # Assume ATR is a feature in the dataset
        transaction_cost = 0.0001  # Fixed transaction cost (example value)

        # Initialize variables
        reward = 0
        profit = 0
        stop_loss_price = None

        if self.position == 1:
            stop_loss_price = self.entry_price - (2 * atr)
        elif self.position == -1:
            stop_loss_price = self.entry_price + (2 * atr)

        # Check if a stop-loss should be triggered using the previous high and low
        if self.position != 0 and stop_loss_price is not None:
            if (self.position == 1 and previous_low <= stop_loss_price) or \
                    (self.position == -1 and previous_high >= stop_loss_price):
                profit = (self.entry_price - current_price) * self.asset_held if self.position == 1 else \
                    (current_price - self.entry_price) * self.asset_held
                self.balance += profit
                self.position = 0  # Reset position
                self.entry_price = 0
                self.asset_held = 0
                reward -= 5  # Penalize for stop-loss trigger

        # Handle actions based on the agent's choice
        if action == 1:  # Buy/Long
            if self.position in [0, -1]:  # If not already long or closing a short
                self.balance -= self.balance * transaction_cost  # Deduct transaction cost
                if self.position == -1:  # Closing a short position
                    diff = self.entry_price - current_price
                    profit = diff * self.asset_held
                    self.balance += profit  # Realize profit/loss
                # Calculate position size to risk 1% of total balance
                stop_loss_price = current_price - (3 * atr)  # Stop-loss is 2ATR below entry price
                stop_loss_distance = abs(current_price - stop_loss_price)
                position_size = (self.balance * 0.03) / stop_loss_distance if stop_loss_distance > 0 else 0
                self.asset_held = position_size
                self.entry_price = current_price
                self.position = 1  # Set position to long

        elif action == 2:  # Sell/Short
            if self.position in [0, 1]:  # If not already short or closing a long
                self.balance -= self.balance * transaction_cost  # Deduct transaction cost
                if self.position == 1:  # Closing a long position
                    diff = current_price - self.entry_price
                    profit = diff * self.asset_held
                    self.balance += profit  # Realize profit/loss
                # Calculate position size to risk 1% of total balance
                stop_loss_price = current_price + (3 * atr)  # Stop-loss is 2ATR above entry price
                stop_loss_distance = abs(current_price - stop_loss_price)
                position_size = (self.balance * 0.03) / stop_loss_distance if stop_loss_distance > 0 else 0
                self.asset_held = position_size
                self.entry_price = current_price
                self.position = -1  # Set position to short

        elif action == 3:  # Close position
            if self.position != 0:
                if self.position == 1:  # Close long position
                    profit = (current_price - self.entry_price) * self.asset_held
                elif self.position == -1:  # Close short position
                    profit = (self.entry_price - current_price) * self.asset_held

                self.balance += profit - self.balance * transaction_cost
                reward += profit  # Adjust reward based on the profit from closing the position
                self.position = 0
                self.entry_price = 0
                self.asset_held = 0

        # Calculate daily return for rewards
        step_return = ((self.balance - previous_balance) / previous_balance
                       if previous_balance != 0 else 0)
        self.step_returns.append(step_return)

        # Rolling metrics (Sharpe ratio and drawdown)
        window_size = min(len(self.data) - 1, len(self.step_returns))
        sharpe_ratio = (np.mean(self.step_returns[-window_size:]) /
                        np.std(self.step_returns[-window_size:])
                        if np.std(self.step_returns[-window_size:]) != 0 else 0)
        drawdown = ((self.max_balance - self.balance) / self.max_balance
                    if self.max_balance != 0 else 0)

        # Update maximum balance observed
        self.max_balance = max(self.max_balance, self.balance)

        # Risk-adjusted reward
        risk_adjusted_reward = profit - (drawdown * 2) + (sharpe_ratio * 0.5)
        reward += risk_adjusted_reward

        # Prepare the next state for the agent
        next_state = self.data.iloc[self.current_step]

        # Additional info for logging and monitoring
        info = {
            'profit': round(profit, 1),
            'balance': round(self.balance, 1),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'drawdown': round(drawdown, 5),
            'stop_loss': stop_loss_price
        }

        # Increment step count
        self.current_step += 1
        return next_state, reward, self.done, info

    def reset(self, scaled_data, orig_data):
        self.current_step = 0
        self.done = False
        self.position = 0
        self.entry_price = 0
        self.balance = self.initial_investment
        self.orig_data = orig_data
        self.data = scaled_data
        return self.data.iloc[self.current_step]

    def render(self, mode='human'):
        pass  # Optional


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def append(self, experience):
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size, alpha=0.6):
        scaled_priorities = np.array(self.priorities) ** alpha
        sample_probs = scaled_priorities / sum(scaled_priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=sample_probs)
        samples = [self.buffer[i] for i in indices]
        return samples, indices

    def update_priorities(self, indices, errors, offset=0.1):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = offset + error
            self.max_priority = max(self.max_priority, error)


# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(50000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.update_target_model()

    def _create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_shape,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.0001))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size, callbacks=[]):
        if len(self.memory.buffer) < batch_size:
            return  # Ensure enough samples are available

        minibatch, indices = self.memory.sample(batch_size)
        states, target_fs = [], []
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            states.append(state[0])  # Accumulate states
            target_fs.append(target_f[0])  # Accumulate target_f values

        # Fit all at once
        self.model.fit(np.array(states), np.array(target_fs), batch_size=batch_size, verbose=0, callbacks=callbacks)
        # Update priorities based on the prediction error, etc.


class TrainingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        _logger.info(f"Epoch {epoch}: Loss: {logs.get('loss')}, Accuracy: {logs.get('accuracy')}")
        # Add other metrics you want to log


class ModelTrainer:

    def __init__(self, zipped_groups, num_of_groups, timeframe):
        _logger.info("Num GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))
        tf.test.gpu_device_name()
        self.checkpoint_filepath = f'model_checkpoint_{timeframe}.weights.h5'
        self.zipp_groups = zipped_groups
        self.num_of_episodes = num_of_groups

        self.logger = TrainingLogger()
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='loss',  # Choose the metric to monitor
            mode='min',
            save_best_only=True)

        self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',  # Choose the metric to monitor
            patience=10,  # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True)

        # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # self.tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        # log_dir_train = "logs/training/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # self.summary_writer = tf.summary.create_file_writer(log_dir_train)

    def train_agent(self, env, agent, resume=True):

        # Load the last checkpoint if resume is True
        if resume and os.path.isfile(self.checkpoint_filepath):
            _logger.info("Resuming from last checkpoint...")
            # agent.model.load_weights(self.checkpoint_filepath)

        total_rewards = []
        total_profits = []
        sharpe_ratios = []
        max_drawdowns = []
        profitable_trades = []
        episode_count = 0

        with tf.device('/GPU:0'):
            for (_, scaled_data), (e_og, orig_data) in self.zipp_groups:
                group_len = len(orig_data)

                state = env.reset(scaled_data, orig_data)
                state = np.reshape(state, [1, env.observation_space.shape[0]])
                total_reward = 0
                total_profit = 0
                episode_sharpe_ratios = []
                episode_max_drawdown = 0
                episode_profitable_trades = 0
                episode_loss_trades = 0

                step_count = 1
                while True:  # Maximum timesteps per episode
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
                    total_reward += reward
                    step_count += 1

                    profit = info['profit']
                    total_profit += profit
                    episode_sharpe_ratios.append(info['sharpe_ratio'])
                    episode_max_drawdown = max(episode_max_drawdown, info['drawdown'])

                    # Track Profitable Trades
                    if profit > 0:
                        episode_profitable_trades += 1
                    if profit < 0:
                        episode_loss_trades += 1

                    agent.remember(state, action, reward, next_state, done)
                    state = next_state

                    agent.replay(group_len * 3, callbacks=[
                        self.model_checkpoint_callback,
                        self.early_stopping_callback,
                        # Other callbacks...
                    ])

                    if done:
                        total_rewards.append(total_reward)
                        total_profits.append(total_profit)
                        sharpe_ratios.append(np.mean(episode_sharpe_ratios))
                        max_drawdowns.append(episode_max_drawdown)
                        profitable_trades.append(episode_profitable_trades)
                        _logger.info(f"END --> Episode: {episode_count}/{self.num_of_episodes}, "
                                     f"balance: {info['balance']}, "
                                     f"Total Reward: {round(total_reward, 1)}, "
                                     f"Total Profit: {round(total_profit, 1)}, "
                                     f"Total Sum profits: {round(sum(total_profits), 1)} "
                                     f"Sharpe Ratio: {round(np.mean(episode_sharpe_ratios), 4)}, "
                                     f"Total Mean Sharpe Ratio: {round(np.mean(sharpe_ratios), 4)}, "
                                     f"Max Drawdown: {round(episode_max_drawdown, 3)}, "
                                     f"Profitable Trades: {episode_profitable_trades} "
                                     f"Loss Trades: {episode_loss_trades}")
                        episode_count += 1
                        break

                    if episode_count % 5 == 0:  # Update target model periodically
                        agent.update_target_model()

            plot_training_performance(total_rewards, 'data/training_plot.png')
        return agent


def plot_training_performance(rewards, save_path=None):
    plt.plot(rewards)
    plt.title('Agent Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        _logger.info(f"Plot saved as {save_path}")

    plt.show()


def predict_new_data(model, new_data):
    new_data = np.reshape(new_data, [1, len(new_data)])
    return np.argmax(model.predict(new_data)[0])


def save_model(model, model_name='trading_model.h5'):
    model.save(model_name)
    _logger.info(f"Model saved as {model_name}")


def load_model(model_name='trading_model.h5'):
    return tf.keras.models.load_model(model_name)


if __name__ == '__main__':
    init_logging()
    length_days = 20
    timeframe = '30m'
    groups = 'day'

    # Assume 'data' is your preprocessed DataFrame
    # data = pd.read_csv(f'data/input_data.csv')
    exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange('bybit')
    exchange_adapter.market = 'BTC'

    data, data_shape_len, groups_len = prepare_training_data(
        exchange_adapter, days=length_days, timeframe=timeframe, group_by=groups
    )

    # Initialize environment and agent
    env = TradingEnv(data_shape_len)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    # Train the agent
    trained_agent = ModelTrainer(data, groups_len, timeframe).train_agent(env, agent)

    # Save the trained model
    save_model(trained_agent.model)

    # Ensure TensorFlow session is properly closed at the end
    tf.keras.backend.clear_session()

    # Later, to make predictions with new data:
    # new_data = ... # your new data as a numpy array
    # loaded_model = load_model()
    # prediction = predict_new_data(loaded_model, new_data)
