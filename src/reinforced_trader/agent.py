import logging
import os
import random
from collections import deque
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
import sqlalchemy.exc
import tensorflow as tf
from gym import spaces
from jnd_utils.log import init_logging
from retrying import retry

from agent_training_data import prepare_training_data
from src.exchange_adapter import BaseExchangeAdapter
from src.exchange_factory import ExchangeFactory
from src.model import trader_database
from src.model.turtle_model import EpisodesTraining

# Set up TensorFlow GPU memory management
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.optimizer.set_jit(True)  # Enable XLA compilation for faster execution
            tf.config.optimizer.set_experimental_options({'layout_optimizer': True})
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

_logger = logging.getLogger(__name__)


# Define the Trading Environment

def load_processed_episodes(suffix):
    with trader_database.session_manager() as session:
        stmt = session.query(EpisodesTraining.episode_group).filter(
            EpisodesTraining.model_suffix == suffix
        ).all()
        result = [row[0] for row in stmt]  # Extract the first element from each tuple
    return result


@retry(retry_on_exception=lambda e: isinstance(e, sqlalchemy.exc.OperationalError),
       stop_max_attempt_number=5,
       wait_exponential_multiplier=1500)
def save_episode(episode, balance, reward, profit, win_trades, loss_trades, suffix):
    # TODO: OperationalError
    _logger.info(f"Saving episode {episode}..")
    with trader_database.session_manager() as session:
        result = EpisodesTraining(
            episode_group=episode,
            balance=balance,
            total_reward=round(reward, 2),
            profit_closed_trades=round(profit, 2),
            win_trades=win_trades,
            lost_trades=loss_trades,
            model_suffix=suffix
        )
        session.add(result)
        session.commit()


def plot_training_performance(rewards, save_path=None):
    try:
        plt.plot(rewards)
        plt.title('Agent Training Performance')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            _logger.info(f"Plot saved as {save_path}")

        plt.show()
    except Exception as exc:
        _logger.error(f"Error plotting performance metrics: {exc}")


def predict_new_data(model, new_data):
    new_data = np.reshape(new_data, [1, len(new_data)])
    return np.argmax(model.predict(new_data)[0])


def save_model(model, model_name='trading_model.h5'):
    model.save(model_name)
    _logger.info(f"Model saved as {model_name}")


def load_model(model_name='trading_model.h5'):
    return tf.keras.models.load_model(model_name)


class TrainingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        _logger.info(f"Epoch {epoch}: Loss: {logs.get('loss')}, Accuracy: {logs.get('accuracy')}")


def create_gru_model(state_shape, action_size):
    model = GRUDQN(state_shape, action_size)

    # Create a dummy input to ensure the model is built
    dummy_input = np.zeros((1, 1, state_shape))
    model(dummy_input)  # This call ensures the model is built

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mean_squared_error']
    )
    return model


class GRUDQN(tf.keras.Model):
    def __init__(self, state_shape, action_size, gru_units=32, dropout_rate=0.2, l2_reg=0.01):
        super(GRUDQN, self).__init__()
        self.gru = tf.keras.layers.GRU(
            gru_units,
            return_sequences=False,
            input_shape=(1, state_shape),
            recurrent_activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.dense1 = tf.keras.layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(
            16,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = tf.keras.layers.Dense(
            action_size,
            activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.dense1(x)
        x = self.dropout1(x, training=True)  # Apply dropout during training
        x = self.dense2(x)
        x = self.dropout2(x, training=True)
        return self.output_layer(x)


class LSTMDQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = create_gru_model(state_shape, action_size)
        self.target_model = create_gru_model(state_shape, action_size)
        self.update_target_model()

    def update_target_model(self):
        """Copy weights to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy action selection."""
        state = np.reshape(state, (1, 1, self.state_shape))  # ✅ Ensure 3D shape for LSTM
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=32, callbacks=[]):
        """Train on past experiences and use callbacks for logging and checkpointing."""
        if len(self.memory) < batch_size:
            return

        # ✅ **Dynamically select minibatch size based on batch_size**
        minibatch_size = min(len(self.memory), batch_size)
        minibatch = random.sample(self.memory, minibatch_size)

        train_batch_size = min(batch_size // 2, len(self.memory))  # ✅ Dynamic batch size

        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, (1, 1, self.state_shape))  # ✅ Fix input shape
            next_state = np.reshape(next_state, (1, 1, self.state_shape))  # ✅ Fix input shape

            if done:
                target = reward
            else:
                next_qs_max = np.amax(self.target_model.predict(next_state, verbose=0)[0])
                target = reward + self.gamma * next_qs_max

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0, batch_size=train_batch_size, callbacks=callbacks)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class TradingEnv(gym.Env):
    """Trading environment for reinforcement learning with cumulative reward calculation."""

    def __init__(self, scaled_data, orig_data, initial_investment=1_000):
        super(TradingEnv, self).__init__()
        self.orig_data = orig_data
        self.data = scaled_data
        self.initial_investment = initial_investment
        self.trade_risk = 1 / 100  # Risk per trade
        self.base_atr_stop_loss_dist = 2  # ATR-based stop-loss distance

        self.current_step = 0
        self.done = False
        self.position = 0  # -1: short, 0: no position, 1: long
        self.entry_price = 0
        self.balance = initial_investment
        self.asset_held = 0
        self.stop_loss = None
        self.max_balance = initial_investment
        self.total_reward = 0  # ✅ Cumulative reward over the entire dataset
        self.total_profit = 0  # ✅ Track cumulative profit

        self.action_space = spaces.Discrete(4)  # 0: hold, 1: buy, 2: sell, 3: close
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(scaled_data.shape[1],),
                                            dtype=np.float32)

    def step(self, action):
        """Executes a trade action and updates the environment with proper reward calculation."""
        current_price = self.orig_data.iloc[self.current_step]['C']
        current_low = self.orig_data.iloc[self.current_step]['L']
        current_high = self.orig_data.iloc[self.current_step]['H']
        atr_lower = self.orig_data.iloc[self.current_step]['atr_20']
        atr_higher = self.orig_data.iloc[self.current_step]['atr_50']
        transaction_cost = 0.055 / 100  # 0.055% per trade

        realized_profit, floating_pnl = 0, 0
        transaction_cost_amount = 0

        # Check stop-loss condition
        if self.position != 0 and self.stop_loss and self.current_step:
            if ((self.position == 1 and current_low <= self.stop_loss) or
                    (self.position == -1 and current_high >= self.stop_loss)):
                realized_profit += self.close_position(self.stop_loss, transaction_cost)

        # Execute trading actions
        if action in [1, 2]:  # Buy (long) or Sell (short)
            new_position = 1 if action == 1 else -1
            if self.position != new_position:
                realized_profit += self.close_position(current_price, transaction_cost)

                stop_loss_distance = self.base_atr_stop_loss_dist * (atr_lower / atr_higher) * atr_lower
                self.stop_loss = current_price - stop_loss_distance if action == 1 else current_price + stop_loss_distance

                max_position_size = self.balance / current_price
                position_size = (self.balance * self.trade_risk) / stop_loss_distance
                position_size = min(max_position_size, position_size)

                transaction_cost_amount = position_size * current_price * transaction_cost
                self.asset_held = position_size
                self.entry_price = current_price
                self.position = new_position
                self.balance -= transaction_cost_amount

        elif action == 3:  # Close position
            realized_profit += self.close_position(current_price, transaction_cost)

        # Calculate Floating PnL
        if self.position != 0:
            floating_pnl = (current_price - self.entry_price) * self.asset_held if self.position == 1 else \
                (self.entry_price - current_price) * self.asset_held

        # Reward Calculation
        step_reward = realized_profit  # - transaction_cost_amount
        step_reward += floating_pnl * 0.1  # Small weight to floating PnL to encourage trend following
        self.total_reward += step_reward  # ✅ Accumulate total episode reward

        # Drawdown Penalty
        self.max_balance = max(self.max_balance, self.balance)
        drawdown = (self.max_balance - self.balance) / self.max_balance if self.max_balance > 0 else 0
        self.total_reward -= drawdown * 0.5  # **Adjust penalty weight (0.5)**

        # Get next state & end condition
        next_state = self.data.iloc[min(self.current_step, len(self.data) - 1)]
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1

        # Step Logging
        _logger.info(f"Step: {self.current_step}, Action: {action}, Position: {self.position}, "
                     f"Current Price: {current_price}, Entry Price: {self.entry_price}, "
                     f"Reward: {round(step_reward, 3)}, Profit: {round(realized_profit, 2)}, "
                     f"Floating PnL: {round(floating_pnl, 2) if self.position != 0 else 0}, "
                     f"Balance: {round(self.balance, 1)}, Drawdown: {round(drawdown, 5)}, Stop Loss: {self.stop_loss}")

        return next_state, step_reward, self.done, {
            'reward': round(step_reward, 3),
            'profit': round(realized_profit, 2),  # ✅ **No total profit here**
            'floating_pnl': round(floating_pnl, 2) if self.position != 0 else 0,
            'balance': round(self.balance, 1),
            'drawdown': round(drawdown, 5),
            'stop_loss': self.stop_loss
        }

    def close_position(self, current_price, transaction_cost):
        """Closes an open position and updates balance."""
        if self.position == 0:
            return 0

        price_diff = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
        profit = price_diff * self.asset_held
        transaction_cost_amount = self.asset_held * current_price * transaction_cost
        actual_profit = profit - transaction_cost_amount

        self.balance += actual_profit
        self.position = 0
        self.entry_price = 0
        self.asset_held = 0
        self.stop_loss = None
        return actual_profit

    def reset(self):
        """Resets the environment for a new episode."""
        self.current_step = 0
        self.done = False
        self.position = 0
        self.entry_price = 0
        self.balance = self.initial_investment
        self.max_balance = self.initial_investment
        self.total_reward = 0  # ✅ Reset cumulative reward at the start of each episode
        self.total_profit = 0  # ✅ Reset cumulative profit at the start of each episode
        return self.data.iloc[self.current_step]


class ModelTrainer:
    def __init__(self, env, agent, model_suffix, resume=True):
        """Initialize training with logging, callbacks, and model checkpointing."""
        _logger.info(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
        tf.test.gpu_device_name()

        self.env = env
        self.agent = agent
        self.model_suffix = model_suffix
        self.checkpoint_filepath = f'models/model_checkpoint_{model_suffix}.weights.h5'

        # Setup logging
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_graph=True
        )

        # Ensure directory exists for model storage
        os.makedirs("models", exist_ok=True)

        # Setup model checkpointing
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True
        )

        # Early stopping
        self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )

        if resume and os.path.isfile(self.checkpoint_filepath):
            _logger.info("Resuming from last checkpoint...")

            # Ensure the dummy input matches the expected input shape
            dummy_input = np.zeros((1, 1, self.env.observation_space.shape[0]))

            # Explicitly call the model with the dummy input to build it
            self.agent.model(dummy_input)  # This should ensure the GRU layer has been built

            # Now load the weights after the model is confirmed to be built
            self.agent.model.load_weights(self.checkpoint_filepath)

    def train_agent(self, num_episodes=500, batch_size=32):
        """Train the agent using LSTM-based DQN on full dataset."""

        with tf.device('/GPU:0'):
            for episode in range(num_episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, 1, self.env.observation_space.shape[0]])  # ✅ Fix input shape

                episode_reward, episode_profit = 0, 0
                episode_sharpe_ratios, episode_max_drawdown = [], 0
                episode_profitable_trades, episode_loss_trades = 0, 0

                while not self.env.done:
                    action = self.agent.act(state)
                    next_state, reward, done, info = self.env.step(action)
                    next_state = np.reshape(next_state,
                                            [1, 1, self.env.observation_space.shape[0]])  # ✅ Fix input shape

                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state

                    episode_reward += reward
                    episode_profit += info['profit']

                    # ✅ Track performance metrics
                    episode_sharpe_ratios.append(info.get('sharpe_ratio', 0))
                    episode_max_drawdown = max(episode_max_drawdown, info.get('drawdown', 0))

                    if info['profit'] > 0:
                        episode_profitable_trades += 1
                    elif info['profit'] < 0:
                        episode_loss_trades += 1

                    # ✅ Replay experience buffer
                    self.agent.replay(batch_size,
                                      callbacks=[
                                          self.tensorboard_callback,
                                          self.model_checkpoint_callback,
                                          self.early_stopping_callback]
                                      )

                    if done:
                        _logger.info(f"Episode {episode}/{num_episodes}, Balance: {info['balance']}, "
                                     f"Total Reward: {round(episode_reward, 1)}, Total Profit: {round(episode_profit, 1)}, "
                                     f"Sharpe Ratio: {round(np.mean(episode_sharpe_ratios), 4)}, "
                                     f"Max Drawdown: {round(episode_max_drawdown, 3)}, "
                                     f"Profitable Trades: {episode_profitable_trades}, Loss Trades: {episode_loss_trades}")

                        save_episode(
                            episode=episode,
                            balance=info['balance'],
                            reward=episode_reward,
                            profit=episode_profit,
                            win_trades=episode_profitable_trades,
                            loss_trades=episode_loss_trades,
                            suffix=self.model_suffix
                        )

                        self.agent.model.save_weights(self.checkpoint_filepath)
                        _logger.info(f"✅ Model checkpoint saved at {self.checkpoint_filepath}")

                        break

                if episode % 5 == 0:
                    self.agent.update_target_model()

        self.save_model()
        return self.agent

    def save_model(self):
        """Save the trained model."""
        model_path = f'models/lstm_trading_model_{self.model_suffix}.h5'
        os.makedirs("models", exist_ok=True)
        self.agent.model.save(model_path)
        _logger.info(f"Model saved as {model_path}")


if __name__ == '__main__':
    # Load dataset
    init_logging()
    ticker, length_days, timeframe = 'BTC', 854, '1d'

    exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange('bybit')
    exchange_adapter.market = ticker

    scaled_data, orig_data = prepare_training_data(
        exchange=exchange_adapter, ticker=ticker, days=length_days, timeframe=timeframe
    )

    # Initialize environment and agent
    env = TradingEnv(scaled_data, orig_data)
    agent = LSTMDQNAgent(env.observation_space.shape[0], env.action_space.n)

    # Train model
    trainer = ModelTrainer(env, agent, model_suffix="BTC_trader")
    trained_agent = trainer.train_agent(num_episodes=5000, batch_size=32)
