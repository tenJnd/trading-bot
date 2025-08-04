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
def save_episode(episode, balance, reward, profit, win_trades, loss_trades, suffix, additional_metrics=None):
    """Enhanced episode saving with additional professional metrics"""

    # Create dict to store episodes
    if not hasattr(save_episode, 'episode_buffer'):
        save_episode.episode_buffer = []

    # Add episode to buffer
    _logger.debug(f"Saving episode {episode} to buffer..")
    save_episode.episode_buffer.append({
        'episode_group': episode,
        'balance': balance,
        'total_reward': round(reward, 2),
        'profit_closed_trades': round(profit, 2),
        'win_trades': win_trades,
        'lost_trades': loss_trades,
        'model_suffix': suffix
    })

    # Save to DB when buffer reaches 100 episodes
    if len(save_episode.episode_buffer) >= 10:
        _logger.info(f"Saving episodes from buffer..")
        trader_database.bulk_insert(EpisodesTraining, save_episode.episode_buffer)

        # Clear buffer after saving
        save_episode.episode_buffer = []


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
    return model.predict(new_data)[0]  # Return raw predictions for multi-discrete


def save_model(model, model_name='trading_model.h5'):
    model.save(model_name)
    _logger.info(f"Model saved as {model_name}")


def load_model(model_name='trading_model.h5'):
    return tf.keras.models.load_model(model_name)


class TrainingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        _logger.info(f"Epoch {epoch}: Loss: {logs.get('loss')}, Accuracy: {logs.get('accuracy')}")


def create_professional_gru_model(state_shape, action_sizes, learning_rate):
    """Enhanced GRU model with larger layers for professional trading"""
    model = ProfessionalGRUDQN(state_shape, action_sizes)

    # Create a dummy input to ensure the model is built
    dummy_input = np.zeros((1, 1, state_shape))
    model(dummy_input)  # This call ensures the model is built

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=['mse', 'mse', 'mse'],  # Simple list - no names needed
        loss_weights=[1.0, 0.5, 0.3]
    )

    return model


class ProfessionalGRUDQN(tf.keras.Model):
    """Enhanced GRU model architecture for professional trading"""

    def __init__(self, state_shape, action_sizes, gru_units=128, dropout_rate=0.3, l2_reg=0.001):
        super(ProfessionalGRUDQN, self).__init__()

        # Multi-layered GRU for better pattern recognition
        self.gru1 = tf.keras.layers.GRU(
            gru_units,
            return_sequences=True,
            input_shape=(1, state_shape),
            recurrent_activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            recurrent_dropout=0.2
        )

        self.gru2 = tf.keras.layers.GRU(
            gru_units // 2,
            return_sequences=False,
            recurrent_activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            recurrent_dropout=0.2
        )

        # Larger dense layers for complex pattern learning
        self.dense1 = tf.keras.layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.dense2 = tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.dense3 = tf.keras.layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

        # Multi-head outputs for multi-discrete actions
        # Direction head: [hold, buy, sell, close, add_to_position]
        self.direction_head = tf.keras.layers.Dense(
            action_sizes[0],
            activation='linear',
            name='direction_output'
        )

        # Position size head: [5%, 10%, 15%, 20%, 25%]
        self.size_head = tf.keras.layers.Dense(
            action_sizes[1],
            activation='linear',
            name='size_output'
        )

        # Confidence head: [low, medium, high]
        self.confidence_head = tf.keras.layers.Dense(
            action_sizes[2],
            activation='linear',
            name='confidence_output'
        )

    def call(self, inputs, training=None):
        x = self.gru1(inputs, training=training)
        x = self.gru2(x, training=training)

        x = self.dense1(x)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        x = self.dense3(x)
        x = self.dropout3(x, training=training)

        # Multi-head outputs
        direction_output = self.direction_head(x)
        size_output = self.size_head(x)
        confidence_output = self.confidence_head(x)

        return [direction_output, size_output, confidence_output]


class ProfessionalDQNAgent:
    """Enhanced DQN Agent for professional trading with multi-discrete actions and smart memory management"""

    def __init__(self, state_shape, action_sizes):
        self.state_shape = state_shape
        self.action_sizes = action_sizes  # [direction, size, confidence]
        self.memory = deque(maxlen=100000)  # Reduced for better quality management
        self.memory_rewards = deque(maxlen=100000)  # Track rewards for quality scoring
        self.memory_timestamps = deque(maxlen=100000)  # Track recency
        self.memory_step = 0  # Step counter for timestamps

        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # Slower decay for better exploration
        self.learning_rate = 0.001
        self.model = create_professional_gru_model(state_shape, action_sizes, self.learning_rate)
        self.target_model = create_professional_gru_model(state_shape, action_sizes, self.learning_rate)
        self.update_target_model()

        # Memory management parameters
        self.memory_cleanup_frequency = 500  # Clean memory every N steps
        self.high_reward_threshold = 5.0  # Threshold for high-value experiences
        self.diversity_action_threshold = 0.1  # Threshold for diverse actions

        # Professional trading parameters
        self.action_names = {
            0: ['hold', 'buy', 'sell', 'close', 'add_to_position'],
            1: ['5%', '10%', '15%', '20%', '25%'],
            2: ['low_conf', 'medium_conf', 'high_conf']
        }

    def update_target_model(self):
        """Copy weights to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def calculate_experience_quality(self, state, action, reward, next_state, done):
        """Calculate quality score for experience"""

        # 1. Reward significance (higher rewards are more valuable)
        reward_score = min(abs(reward) / 10.0, 1.0)  # Normalize and cap at 1.0

        # 2. Action diversity (non-hold actions are more valuable)
        action_array = np.array(action)
        action_diversity = 1.0 if np.sum(action_array) > self.diversity_action_threshold else 0.3

        # 3. State transition significance (bigger changes are more informative)
        if state is not None and next_state is not None:
            state_flat = np.array(state).flatten()
            next_state_flat = np.array(next_state).flatten()
            state_change = np.sum(np.abs(next_state_flat - state_flat))
            state_significance = min(state_change / 100.0, 1.0)
        else:
            state_significance = 0.5  # Default for missing states

        # 4. Terminal state bonus (episode endings are important)
        terminal_bonus = 1.0 if done else 0.5

        # Combine scores with weights
        quality_score = (
                0.4 * reward_score +
                0.2 * action_diversity +
                0.2 * state_significance +
                0.2 * terminal_bonus
        )

        return min(quality_score, 1.0)  # Cap at 1.0

    def remember(self, state, action, reward, next_state, done):
        """Store experience with quality scoring and smart memory management"""

        # Calculate quality score for this experience
        quality_score = self.calculate_experience_quality(state, action, reward, next_state, done)

        # Store experience with metadata
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        self.memory_rewards.append(abs(reward))  # Store absolute reward for analysis
        self.memory_timestamps.append(self.memory_step)

        self.memory_step += 1

        # Periodic memory cleanup
        if self.memory_step % self.memory_cleanup_frequency == 0:
            self._optimize_memory()

    def _optimize_memory(self):
        """Smart memory optimization keeping high-quality experiences"""

        if len(self.memory) < 50000:  # Only optimize when memory is substantial
            return

        _logger.info(f"Optimizing memory buffer: {len(self.memory)} experiences")

        # Convert to lists for processing
        experiences = list(self.memory)
        rewards = list(self.memory_rewards)
        timestamps = list(self.memory_timestamps)

        # Categorize experiences
        high_reward_indices = []
        recent_indices = []
        diverse_action_indices = []

        current_step = self.memory_step
        recent_threshold = current_step - 15000  # Last 15k steps

        for i, (exp, reward, timestamp) in enumerate(zip(experiences, rewards, timestamps)):
            # High reward experiences
            if reward > self.high_reward_threshold:
                high_reward_indices.append(i)

            # Recent experiences
            if timestamp > recent_threshold:
                recent_indices.append(i)

            # Diverse actions (non-hold actions)
            action = exp[1]
            if np.sum(np.array(action)) > self.diversity_action_threshold:
                diverse_action_indices.append(i)

        # Combine categories (with overlap allowed)
        priority_indices = set(high_reward_indices + recent_indices + diverse_action_indices)

        # If we still have too many, prioritize recent experiences
        if len(priority_indices) > 40000:
            priority_indices = set(recent_indices[-30000:] + high_reward_indices[:10000])

        # Create new optimized memory
        optimized_experiences = [experiences[i] for i in sorted(priority_indices)]
        optimized_rewards = [rewards[i] for i in sorted(priority_indices)]
        optimized_timestamps = [timestamps[i] for i in sorted(priority_indices)]

        # Shuffle to avoid temporal bias
        combined = list(zip(optimized_experiences, optimized_rewards, optimized_timestamps))
        np.random.shuffle(combined)
        optimized_experiences, optimized_rewards, optimized_timestamps = zip(*combined)

        # Replace memory with optimized version
        self.memory = deque(optimized_experiences, maxlen=100000)
        self.memory_rewards = deque(optimized_rewards, maxlen=100000)
        self.memory_timestamps = deque(optimized_timestamps, maxlen=100000)

        _logger.info(f"Memory optimized: {len(self.memory)} quality experiences retained")
        _logger.info(f"  - High reward: {len(high_reward_indices)}")
        _logger.info(f"  - Recent: {len(recent_indices)}")
        _logger.info(f"  - Diverse actions: {len(diverse_action_indices)}")

    def act(self, state):
        """Enhanced epsilon-greedy action selection for multi-discrete actions"""
        state = np.reshape(state, (1, 1, self.state_shape))

        if np.random.rand() <= self.epsilon:
            # Random exploration with some trading logic
            direction = random.randrange(self.action_sizes[0])
            size = random.randrange(self.action_sizes[1])
            confidence = random.randrange(self.action_sizes[2])
            return [direction, size, confidence]

        # Get Q-values for each action head
        q_values = self.model.predict(state, verbose=0)

        # Select best action for each head
        direction = np.argmax(q_values[0][0])
        size = np.argmax(q_values[1][0])
        confidence = np.argmax(q_values[2][0])

        return [direction, size, confidence]

    def replay(self, batch_size=64, callbacks=[]):
        """Enhanced training with multi-discrete action support and prioritized sampling"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        # Prepare batch data
        states = np.array([np.reshape(e[0], (1, self.state_shape)) for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([np.reshape(e[3], (1, self.state_shape)) for e in minibatch])
        dones = np.array([e[4] for e in minibatch])

        # Reshape for batch processing
        batch_size = len(minibatch)
        states = states.reshape((batch_size, 1, self.state_shape))
        next_states = next_states.reshape((batch_size, 1, self.state_shape))

        # Get current Q-values
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q-values for each head
        target_direction = current_q_values[0].copy()
        target_size = current_q_values[1].copy()
        target_confidence = current_q_values[2].copy()

        for i in range(batch_size):
            if dones[i]:
                target_value = rewards[i]
            else:
                # Use max Q-value from next state for each head
                next_direction_max = np.max(next_q_values[0][i])
                next_size_max = np.max(next_q_values[1][i])
                next_confidence_max = np.max(next_q_values[2][i])

                # Average the Q-values from all heads for more stable learning
                target_value = rewards[i] + self.gamma * (
                        next_direction_max + next_size_max + next_confidence_max
                ) / 3

            # Update targets for taken actions
            target_direction[i][actions[i][0]] = target_value
            target_size[i][actions[i][1]] = target_value
            target_confidence[i][actions[i][2]] = target_value

        # Train the model with SIMPLE LIST FORMAT
        try:
            history = self.model.fit(
                states,
                [target_direction, target_size, target_confidence],  # Simple list
                epochs=1,
                verbose=0,
                batch_size=min(32, batch_size),
                callbacks=callbacks
            )

            # Decay epsilon after successful training
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            return history

        except Exception as e:
            _logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return None


class ProfessionalTradingEnv(gym.Env):
    """Enhanced trading environment with professional features"""

    def __init__(
            self,
            data,
            not_scaled_data,
            episode_length_base=168,
            episode_extension=50,
            initial_investment=10_000
    ):
        super(ProfessionalTradingEnv, self).__init__()
        # Store both scaled and original data
        self.data = data  # scaled_data
        self.orig_data = not_scaled_data
        self.full_data = data  # Keep reference to full dataset
        self.full_orig_data = not_scaled_data  # Keep reference to full original dataset

        # Episode management
        self.episode_length_base = episode_length_base  # Base episode length
        self.episode_extension = episode_extension  # Extension steps
        self.total_episode_length = self.episode_length_base + self.episode_extension  # 218 total
        self.episode_start_idx = 0  # Start index of current episode

        self.initial_investment = initial_investment

        # Professional trading parameters
        self.base_risk_levels = [0.005, 0.01, 0.015, 0.02, 0.025]  # Risk per position size
        self.confidence_multipliers = [0.7, 1.0, 1.3]  # Risk adjustment based on confidence
        self.position_size_percentages = [0.05, 0.10, 0.15, 0.20, 0.25]  # Position sizes

        # Pyramid trading parameters
        self.max_pyramid_levels = 3
        self.pyramid_threshold = 0.015  # 1.5% favorable move required
        self.pyramid_entries = []  # Track pyramid entries

        # Enhanced fee structure (more realistic)
        self.maker_fee = 0.02 / 100  # 0.02% maker fee
        self.taker_fee = 0.055 / 100  # 0.055% taker fee
        self.funding_rate = 0.01 / 100  # 0.01% funding rate (approximate)

        # State tracking
        self.current_step = 0
        self.done = False
        self.position = 0  # -1: short, 0: no position, 1: long
        self.entry_price = 0
        self.balance = initial_investment
        self.asset_held = 0
        self.stop_loss = None
        self.take_profit = None

        # Performance tracking
        self.max_balance = initial_investment
        self.total_reward = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        self.max_drawdown = 0
        self.trade_history = []

        # Enhanced action space: [direction, position_size, confidence]
        # Direction: 0=hold, 1=buy, 2=sell, 3=close, 4=add_to_position
        # Position size: 0=5%, 1=10%, 2=15%, 3=20%, 4=25%
        # Confidence: 0=low, 1=medium, 2=high
        self.action_space = spaces.MultiDiscrete([5, 5, 3])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(scaled_data.shape[1],),
            dtype=np.float32
        )

    def calculate_position_size(self, action, current_price, atr):
        """Calculate professional position size based on action parameters"""
        direction, size_idx, confidence_idx = action

        if direction == 0:  # Hold
            return 0

        # Base risk and size
        base_risk = self.base_risk_levels[size_idx]
        confidence_mult = self.confidence_multipliers[confidence_idx]

        # Market condition adjustments
        volatility_adj = self.get_volatility_adjustment(atr)
        trend_strength_adj = self.get_trend_strength_adjustment()
        drawdown_adj = self.get_drawdown_adjustment()

        # Calculate final risk
        final_risk = base_risk * confidence_mult * volatility_adj * trend_strength_adj * drawdown_adj
        final_risk = np.clip(final_risk, 0.002, 0.05)  # 0.2% to 5% risk bounds

        # Position size calculation
        stop_distance = 2 * atr  # 2x ATR stop
        position_value = self.balance * final_risk
        position_size = position_value / stop_distance

        # Ensure we don't exceed available balance
        max_position_size = (self.balance * 0.95) / current_price  # 95% of balance max
        position_size = min(position_size, max_position_size)

        return position_size

    def get_volatility_adjustment(self, current_atr):
        """Adjust position size based on volatility"""
        if self.current_step < 20:
            return 1.0

        recent_atr = self.orig_data['atr_20'].iloc[max(0, self.current_step - 20):self.current_step].mean()
        if recent_atr == 0:
            return 1.0

        volatility_ratio = current_atr / recent_atr
        # Reduce position size in high volatility
        return max(0.5, min(1.5, 1.0 / np.sqrt(volatility_ratio)))

    def get_trend_strength_adjustment(self):
        """Adjust position size based on trend strength"""
        if self.current_step < 20:
            return 1.0

        current_price = self.orig_data.iloc[self.current_step]['C']
        sma_20 = self.orig_data.iloc[max(0, self.current_step - 20):self.current_step]['C'].mean()

        # Simple trend strength indicator
        trend_strength = abs(current_price - sma_20) / sma_20
        return min(1.3, 1.0 + trend_strength * 2)  # Max 30% increase

    def get_drawdown_adjustment(self):
        """Reduce position size during drawdown periods"""
        current_drawdown = (self.max_balance - self.balance) / self.max_balance
        if current_drawdown < 0.05:  # Less than 5% drawdown
            return 1.0
        elif current_drawdown < 0.15:  # 5-15% drawdown
            return 0.8
        else:  # More than 15% drawdown
            return 0.5

    def can_pyramid(self, current_price):
        """Check if pyramiding is allowed"""
        if len(self.pyramid_entries) >= self.max_pyramid_levels:
            return False

        if self.position == 0:
            return False

        # Check if position is profitable enough
        if self.position > 0:  # Long position
            profit_pct = (current_price - self.entry_price) / self.entry_price
        else:  # Short position
            profit_pct = (self.entry_price - current_price) / self.entry_price

        return profit_pct >= self.pyramid_threshold

    def add_to_position(self, action, current_price, atr):
        """Add to existing position (pyramid entry)"""
        if not self.can_pyramid(current_price):
            return 0, 0

        # Calculate pyramid size (smaller than original)
        pyramid_multiplier = 0.6 ** len(self.pyramid_entries)  # Decreasing size
        pyramid_size = self.calculate_position_size(action, current_price, atr) * pyramid_multiplier

        if pyramid_size < 0.01:  # Minimum size check
            return 0, 0

        # Update weighted average entry price
        total_cost = (self.asset_held * self.entry_price) + (pyramid_size * current_price)
        total_shares = self.asset_held + pyramid_size

        if total_shares > 0:
            self.entry_price = total_cost / total_shares
            self.asset_held = total_shares

            # Transaction cost
            transaction_cost = pyramid_size * current_price * self.taker_fee
            self.balance -= transaction_cost

            # Track pyramid entry
            self.pyramid_entries.append({
                'price': current_price,
                'size': pyramid_size,
                'step': self.current_step
            })

            return pyramid_size, transaction_cost

        return 0, 0

    def calculate_fees(self, position_size, current_price, is_maker=False):
        """Calculate realistic trading fees"""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        trading_fee = position_size * current_price * fee_rate

        # Add funding cost for held positions (simplified)
        funding_cost = 0
        if self.position != 0 and self.asset_held > 0:
            funding_cost = self.asset_held * current_price * self.funding_rate

        return trading_fee + funding_cost

    def _should_end_episode(self):
        """Determine if episode should end based on extension logic"""
        # Always end if we've reached the absolute data limit
        if self.current_step >= len(self.data):
            return True

        # End if we've passed the base episode length AND have no open position
        if self.current_step >= self.episode_length_base and self.position == 0:
            return True

        # Force end if we've used all extension steps
        if self.current_step >= self.total_episode_length:
            return True

        return False

    def step(self, action):
        """Enhanced step function with professional trading logic"""
        current_price = self.orig_data.iloc[self.current_step]['C']
        current_low = self.orig_data.iloc[self.current_step]['L']
        current_high = self.orig_data.iloc[self.current_step]['H']
        atr = self.orig_data.iloc[self.current_step]['atr_20']

        direction, size_idx, confidence_idx = action
        realized_profit = 0
        transaction_costs = 0

        # # Check stop-loss and take-profit
        # if self.position != 0 and self.stop_loss:
        #     if ((self.position == 1 and current_low <= self.stop_loss) or
        #             (self.position == -1 and current_high >= self.stop_loss)):
        #         realized_profit += self.close_position(self.stop_loss)

        # Execute trading actions
        if direction == 0:  # Hold
            pass

        elif direction == 1:  # Buy (long)
            if self.position <= 0:
                if self.position < 0:  # Close short first
                    realized_profit += self.close_position(current_price)

                # Open new long position
                position_size = self.calculate_position_size(action, current_price, atr)
                if position_size > 0:
                    self.open_position(1, current_price, position_size, atr)
                    transaction_costs += self.calculate_fees(position_size, current_price)

        elif direction == 2:  # Sell (short)
            if self.position >= 0:
                if self.position > 0:  # Close long first
                    realized_profit += self.close_position(current_price)

                # Open new short position
                position_size = self.calculate_position_size(action, current_price, atr)
                if position_size > 0:
                    self.open_position(-1, current_price, position_size, atr)
                    transaction_costs += self.calculate_fees(position_size, current_price)

        elif direction == 3:  # Close position
            if self.position != 0:
                realized_profit += self.close_position(current_price)

        elif direction == 4:  # Add to position
            if self.position != 0:
                pyramid_size, pyramid_cost = self.add_to_position(action, current_price, atr)
                transaction_costs += pyramid_cost

        # Calculate floating PnL
        floating_pnl = 0
        if self.position != 0:
            if self.position == 1:  # Long
                floating_pnl = (current_price - self.entry_price) * self.asset_held
            else:  # Short
                floating_pnl = (self.entry_price - current_price) * self.asset_held

        # Enhanced reward calculation
        step_reward = self.calculate_reward_profit_only(
            realized_profit, floating_pnl, transaction_costs, action
        )

        # Update performance metrics
        self.update_performance_metrics(realized_profit)

        # Get next state
        next_state = self.data.iloc[min(self.current_step, len(self.data) - 1)]
        self.current_step += 1
        self.done = self._should_end_episode()

        # Enhanced logging
        action_desc = f"{self.get_action_description(action)}"
        # _logger.info(
        #     f"Step: {self.current_step}, Action: {action_desc}, "
        #     f"Position: {self.position}, Price: ${current_price:.2f}, "
        #     f"Reward: {step_reward:.3f}, Balance: ${self.balance:.2f}, "
        #     f"Floating P&L: ${floating_pnl:.2f} "
        # )

        info = {
            'reward': step_reward,
            'realized_profit': realized_profit,
            'floating_pnl': floating_pnl,
            'balance': self.balance,
            'position': self.position,
            'transaction_costs': transaction_costs,
            'pyramid_levels': len(self.pyramid_entries),
            'action_description': action_desc,
            'current_step': self.current_step,
            'base_length_reached': self.current_step >= self.episode_length_base,
            'extension_steps_used': max(0, self.current_step - self.episode_length_base),
            'can_end_naturally': self.current_step >= self.episode_length_base and self.position == 0,

        }

        return next_state, step_reward, self.done, info

    def calculate_reward(self, realized_profit, floating_pnl, transaction_costs, action):
        """Profit-focused reward function"""
        direction, size_idx, confidence_idx = action

        # CORE: Actual profit after costs (this should dominate)
        net_profit = realized_profit - transaction_costs
        step_reward = net_profit * 10  # Scale up the importance of actual profit

        # Only add bonuses if we're actually making money
        if net_profit > 0:
            # Reward appropriate sizing on winning trades
            size_bonus = self.calculate_sizing_bonus(size_idx, confidence_idx)
            step_reward += size_bonus

            # Risk management bonus (only on profitable trades)
            risk_reward = self.calculate_risk_management_reward() * 0.1  # Reduced weight
            step_reward += risk_reward

            # Small floating PnL contribution
            step_reward += floating_pnl * 0.02

        else:  # Losing trade
            # Heavy penalties for losses
            size_penalty = self.calculate_sizing_penalty(size_idx, abs(net_profit))
            step_reward -= size_penalty * 1.2  # penalty

            # Drawdown penalty (more severe)
            drawdown_penalty = self.calculate_drawdown_penalty() * 1.2
            step_reward -= drawdown_penalty

            # Negative contribution from floating losses
            step_reward += floating_pnl * 0.1  # This will be negative

        # Episode-end adjustment: Heavy penalty if final balance < initial
        if hasattr(self, 'done') and self.done:
            final_return = (self.balance - self.initial_investment) / self.initial_investment
            if final_return < 0:
                # HUGE penalty for losing money overall
                step_reward -= abs(final_return) * 1000  # Scale this appropriately
            else:
                # Bonus for profitable episodes
                step_reward += final_return * 500

        self.total_reward += step_reward
        return step_reward

    def calculate_reward_profit_only(self, realized_profit, floating_pnl, transaction_costs, action):
        """Simple profit-focused reward"""

        # The ONLY thing that matters: actual profit
        net_profit = realized_profit - transaction_costs

        # Base reward is just the profit/loss
        step_reward = net_profit

        # Small adjustment for unrealized PnL
        step_reward += floating_pnl * 0.01

        # Episode end: BIG adjustment based on total performance
        if hasattr(self, 'done') and self.done:
            total_return = (self.balance - self.initial_investment)
            # This dominates everything else
            step_reward += total_return  # Adjust multiplier as needed

        self.total_reward += step_reward
        return step_reward

    def calculate_sizing_bonus(self, size_idx, confidence_idx):
        """Reward appropriate position sizing"""
        # Higher confidence with larger size = bonus
        if confidence_idx == 2 and size_idx >= 3:  # High confidence, large size
            return 5.0
        elif confidence_idx == 1 and size_idx == 2:  # Medium confidence, medium size
            return 2.0
        elif confidence_idx == 0 and size_idx <= 1:  # Low confidence, small size
            return 1.0
        return 0

    def calculate_sizing_penalty(self, size_idx, loss_amount):
        """Penalize oversizing on losing trades"""
        if size_idx >= 3 and loss_amount > self.balance * 0.02:  # Large size, big loss
            return loss_amount * 0.5
        return 0

    def calculate_risk_management_reward(self):
        """Reward good risk management practices"""
        reward = 0

        # Reward for using stop losses
        if self.stop_loss is not None:
            reward += 1.0

        # Reward for pyramid trading when profitable
        if len(self.pyramid_entries) > 0 and self.position != 0:
            current_price = self.orig_data.iloc[self.current_step]['C']
            if ((self.position == 1 and current_price > self.entry_price) or
                    (self.position == -1 and current_price < self.entry_price)):
                reward += len(self.pyramid_entries) * 2.0

        return reward

    def calculate_consistency_bonus(self):
        """Reward consistent performance"""
        if len(self.trade_history) < 5:
            return 0

        recent_trades = self.trade_history[-5:]
        winning_ratio = sum(1 for trade in recent_trades if trade > 0) / len(recent_trades)

        if winning_ratio >= 0.6:  # 60% win rate
            return 10.0
        elif winning_ratio >= 0.4:  # 40% win rate
            return 5.0
        return 0

    def calculate_drawdown_penalty(self):
        """Penalize excessive drawdowns"""
        current_drawdown = (self.max_balance - self.balance) / self.max_balance
        if current_drawdown > 0.15:  # More than 15% drawdown
            return current_drawdown * 50
        elif current_drawdown > 0.1:  # More than 10% drawdown
            return current_drawdown * 20
        return 0

    def open_position(self, direction, current_price, position_size, atr):
        """Open a new position with enhanced logic"""
        self.position = direction
        self.entry_price = current_price
        self.asset_held = position_size

        # Set stop loss and take profit
        stop_distance = 5 * atr
        self.stop_loss = (current_price - stop_distance if direction == 1
                          else current_price + stop_distance)

        # Take profit at 2:1 risk-reward ratio
        profit_distance = stop_distance * 10
        self.take_profit = (current_price + profit_distance if direction == 1
                            else current_price - profit_distance)

        # Reset pyramid tracking
        self.pyramid_entries = []

    def close_position(self, current_price):
        """Close position with enhanced tracking"""
        if self.position == 0:
            return 0

        # Calculate profit
        if self.position == 1:  # Long
            profit = (current_price - self.entry_price) * self.asset_held
        else:  # Short
            profit = (self.entry_price - current_price) * self.asset_held

        # Transaction costs
        transaction_cost = self.calculate_fees(self.asset_held, current_price)
        net_profit = profit - transaction_cost

        # Update balance
        self.balance += net_profit

        # Track trade
        self.trade_history.append(net_profit)
        self.total_trades += 1

        if net_profit > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1

        # Reset position
        self.position = 0
        self.entry_price = 0
        self.asset_held = 0
        self.stop_loss = None
        self.take_profit = None
        self.pyramid_entries = []

        return net_profit

    def update_performance_metrics(self, realized_profit):
        """Update various performance metrics"""
        self.max_balance = max(self.max_balance, self.balance)
        current_drawdown = (self.max_balance - self.balance) / self.max_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

    @staticmethod
    def get_action_description(action):
        """Get human-readable action description"""
        direction, size_idx, confidence_idx = action
        direction_name = ['Hold', 'Buy', 'Sell', 'Close', 'Add'][direction]
        size_name = ['5%', '10%', '15%', '20%', '25%'][size_idx]
        confidence_name = ['Low', 'Med', 'High'][confidence_idx]
        return f"{direction_name}({size_name},{confidence_name})"

    def reset(self):
        """Reset environment for new episode"""
        # Calculate available starting positions
        total_data_length = len(self.full_data)

        if total_data_length < self.total_episode_length:
            # If not enough data, use all available data
            self.episode_start_idx = 0
            self.data = self.full_data
            self.orig_data = self.full_orig_data
        else:
            # Random sampling: pick random start position
            max_start_idx = total_data_length - self.total_episode_length
            self.episode_start_idx = np.random.randint(0, max_start_idx + 1)

            # Slice the data for this episode (base + extension)
            episode_end_idx = self.episode_start_idx + self.total_episode_length
            self.data = self.full_data[self.episode_start_idx:episode_end_idx]
            self.orig_data = self.full_orig_data[self.episode_start_idx:episode_end_idx]

        self.current_step = 0
        self.done = False
        self.position = 0
        self.entry_price = 0
        self.balance = self.initial_investment
        self.asset_held = 0
        self.stop_loss = None
        self.take_profit = None
        self.max_balance = self.initial_investment
        self.total_reward = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        self.max_drawdown = 0
        self.trade_history = []
        self.pyramid_entries = []

        return self.data.iloc[self.current_step]


class ProfessionalModelTrainer:
    """Enhanced model trainer for professional trading"""
    TARGET_UPDATE_FREQUENCY = 1000  # Update every 1000 training steps

    def __init__(self, env, agent, model_suffix, resume=True):
        _logger.info(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

        self.env = env
        self.agent = agent
        self.model_suffix = model_suffix
        self.training_steps = 0
        self.checkpoint_filepath = f'checkpoints/professional_model_{model_suffix}.weights.h5'

        # Enhanced logging
        log_dir = "logs/professional_trading/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq='epoch'
        )

        os.makedirs("models", exist_ok=True)

        # Enhanced callbacks
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            monitor='loss',  # Make sure this matches
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=100,
            restore_best_weights=True,
            verbose=1

        )

        self.reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.8,
            patience=50,
            min_lr=0.00001,
            verbose=1
        )

        # Resume from checkpoint
        if resume and os.path.isfile(self.checkpoint_filepath):
            _logger.info("Resuming from professional model checkpoint...")
            dummy_input = np.zeros((1, 1, self.env.observation_space.shape[0]))
            self.agent.model(dummy_input)
            self.agent.model.load_weights(self.checkpoint_filepath)

    def train_agent(self, num_episodes=5000, batch_size=128):
        """Enhanced training with professional metrics tracking"""
        best_balance = 0
        training_frequency = 12  # Train every N steps
        step_counter = 0
        early_stopping_patience = 300

        episode_rewards = []
        episode_balances = []

        with tf.device('/GPU:0'):
            for episode in range(num_episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, 1, self.env.observation_space.shape[0]])

                episode_reward = 0
                episode_metrics = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'max_drawdown': 0,
                    'pyramid_entries': 0,
                    'transaction_costs': 0
                }

                # EPISODES STEPS ###
                while not self.env.done:
                    action = self.agent.act(state)
                    next_state, reward, done, info = self.env.step(action)
                    next_state = np.reshape(next_state, [1, 1, self.env.observation_space.shape[0]])

                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward

                    step_counter += 1

                    # Update episode metrics
                    episode_metrics['transaction_costs'] += info.get('transaction_costs', 0)
                    episode_metrics['pyramid_entries'] = max(
                        episode_metrics['pyramid_entries'],
                        info.get('pyramid_levels', 0)
                    )

                    if (step_counter % training_frequency == 0 and
                            len(self.agent.memory) > batch_size):
                        history = self.agent.replay(batch_size=batch_size)
                        self.training_steps += 1

                        # Update target model based on training steps
                        if self.training_steps % self.TARGET_UPDATE_FREQUENCY == 0:
                            self.agent.update_target_model()
                            _logger.info(f"Target model updated at training step {self.training_steps}")
                #######################################################

                # Episode is complete ###
                # calculate final episode metrics
                win_rate = (self.env.winning_trades / max(self.env.total_trades, 1)) * 100
                profit_factor = self.calculate_profit_factor()
                sharpe_ratio = self.calculate_sharpe_ratio()

                episode_rewards.append(episode_reward)
                episode_balances.append(info['balance'])

                _logger.info(
                    f"Episode {episode}/{num_episodes} | "
                    f"Balance: ${info['balance']:.2f} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Trades: {self.env.total_trades} | "
                    f"Win Rate: {win_rate:.1f}% | "
                    f"Max DD: {self.env.max_drawdown:.3f} | "
                    f"Sharpe: {sharpe_ratio:.3f} | "
                    f"PF: {profit_factor:.2f} | "
                    f"Epsilon: {self.agent.epsilon:.4f} | "
                    f"Memory after episode: {len(self.agent.memory)}"
                )

                # EPISODE SAVE #
                # Save episode with enhanced metrics
                save_episode(
                    episode=episode,
                    balance=info['balance'],
                    reward=episode_reward,
                    profit=sum([t for t in self.env.trade_history if t > 0]),
                    win_trades=self.env.winning_trades,
                    loss_trades=self.env.losing_trades,
                    suffix=self.model_suffix,
                    additional_metrics={
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': self.env.max_drawdown
                    }
                )

                # CHECKPOINTS AND EARLY STOPPING #
                # Manual checkpointing based on performance only
                current_balance = self.env.balance
                if current_balance > best_balance:
                    best_balance = current_balance
                    checkpoint_path = f"checkpoints/best_model_episode.keras"
                    self.save_model(checkpoint_path)
                    patience_counter = 0  # Reset patience
                    _logger.info(f"New best balance: ${current_balance:.2f} - Model saved to {checkpoint_path}")
                else:
                    patience_counter += 1

                # if patience_counter >= early_stopping_patience:
                #     _logger.info(f"Early stopping triggered after {patience_counter} episodes without improvement")
                #     break

        self.save_model()
        return self.agent

    def calculate_profit_factor(self):
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.env.trade_history:
            return 0

        gross_profit = sum([trade for trade in self.env.trade_history if trade > 0])
        gross_loss = abs(sum([trade for trade in self.env.trade_history if trade < 0]))

        return gross_profit / max(gross_loss, 1)

    def calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio of returns"""
        if len(self.env.trade_history) < 2:
            return 0

        returns = np.array(self.env.trade_history)
        return np.mean(returns) / (np.std(returns) + 1e-6)

    def save_model(self, model_path=None):
        """Save the professional trading model"""
        if not model_path:
            model_path = f'models/professional_trading_model_{self.model_suffix}.keras'
            os.makedirs("models", exist_ok=True)

        self.agent.model.save(model_path)
        _logger.info(f"Professional model saved as {model_path}")


if __name__ == '__main__':
    # Initialize professional trading system
    init_logging()
    ticker, length_days, timeframe = 'BTC', 854, '1d'

    exchange_adapter: BaseExchangeAdapter = ExchangeFactory.get_exchange('bybit')
    exchange_adapter.market = ticker

    scaled_data, orig_data = prepare_training_data(
        exchange=exchange_adapter, ticker=ticker, days=length_days, timeframe=timeframe
    )

    # Initialize professional environment and agent
    env = ProfessionalTradingEnv(
        data=scaled_data,
        not_scaled_data=orig_data,
        episode_length_base=14,
        episode_extension=7,
        initial_investment=10_000
    )

    # Multi-discrete action sizes: [direction, position_size, confidence]
    action_sizes = [5, 5, 3]  # [hold/buy/sell/close/add, 5%-25%, low/med/high]

    agent = ProfessionalDQNAgent(env.observation_space.shape[0], action_sizes)

    # Train professional model
    trainer = ProfessionalModelTrainer(env, agent, model_suffix="BTC_professional_trader")
    trained_agent = trainer.train_agent(num_episodes=10000, batch_size=128)
