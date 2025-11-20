import gymnasium as gym
import numpy as np
import pickle
from collections import defaultdict


class QLearningAgent:
    def __init__(self, action_space_size, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.01):
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_space_size = action_space_size

    def discretize_state(self, observation):
        """Спрощення стану до дискретного представлення"""

        # ROI - область попереду машини
        roi_top = 40
        roi_bottom = 65
        roi_left = 25
        roi_right = 71

        roi = observation[roi_top:roi_bottom, roi_left:roi_right]

        # Перетворюємо в grayscale
        gray = np.mean(roi, axis=2)

        # Ділимо на 3 зони
        width = gray.shape[1]
        zone1_end = width // 3
        zone2_end = 2 * width // 3

        left = np.mean(gray[:, :zone1_end])
        center = np.mean(gray[:, zone1_end:zone2_end])
        right = np.mean(gray[:, zone2_end:])

        # Класифікація (дорога темніша ~100-110, трава світліша ~120-135)
        def classify(val):
            if val < 110:
                return 2  # темне = дорога (сіра асфальт)
            elif val < 120:
                return 1  # середнє = край дороги або тінь
            else:
                return 0  # світле = трава (зелена яскрава)

        return (classify(left), classify(center), classify(right))

    def choose_action(self, state):
        """Epsilon-greedy вибір дії"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """Оновлення Q-значень"""
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        # Q-learning update rule
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Зменшення epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        """Зберегти модель"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'lr': self.lr,
                'gamma': self.gamma
            }, f)
        print(f"Модель збережено у {filename}")

    def load(self, filename):
        """Завантажити модель"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size), data['q_table'])
            self.epsilon = data['epsilon']
            self.lr = data['lr']
            self.gamma = data['gamma']
        print(f"Модель завантажено з {filename}")


import matplotlib.pyplot as plt
from IPython import display

import matplotlib.pyplot as plt

import cv2
import numpy as np


def train_q_learning_opencv(episodes=500, max_steps=1000, save_path="q_learning_car_racing.pkl"):
    """Тренування з візуалізацією через OpenCV (швидко!)"""

    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95,
                   domain_randomize=False, continuous=False)

    agent = QLearningAgent(action_space_size=5)

    episode_rewards = []

    # Створюємо вікно OpenCV
    cv2.namedWindow('Car Racing Training', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Car Racing Training', 800, 600)

    for episode in range(episodes):
        observation, info = env.reset(seed=42)
        state = agent.discretize_state(observation)

        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_observation)

            # Модифікуємо винагороду
            if reward < 0:
                reward = -10
            else:
                reward += 0.5

            if action == 3:
                reward += 30
            elif action == 0:
                reward -= 0.2
            elif action == 4:
                reward -= 0.1

            agent.update(state, action, reward, next_state, terminated or truncated)

            # Візуалізація через OpenCV (ШВИДКО!)
            frame = env.render()

            # Конвертуємо RGB -> BGR для OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Додаємо текст на кадр
            action_names = ['Нічого', 'Ліво', 'Право', 'Газ', 'Гальмо']
            cv2.putText(frame_bgr, f'Episode: {episode}/{episodes}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f'Step: {step}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f'Action: {action_names[action]}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f'Reward: {total_reward:.1f}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f'Epsilon: {agent.epsilon:.3f}', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Показуємо кадр
            cv2.imshow('Car Racing Training', frame_bgr)

            # Перевірка натискання клавіші (ESC для виходу)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                cv2.destroyAllWindows()
                env.close()
                agent.save(save_path)
                return agent, episode_rewards

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    cv2.destroyAllWindows()
    env.close()
    agent.save(save_path)

    return agent, episode_rewards

def test_agent(model_path="q_learning_car_racing.pkl", episodes=5):
    """Тестування навченого агента"""
    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95,
                   domain_randomize=False, continuous=False)

    agent = QLearningAgent(action_space_size=env.action_space.n)
    agent.load(model_path)
    agent.epsilon = 0  # Вимкнути exploration під час тестування

    for episode in range(episodes):
        observation, info = env.reset(seed=42)
        state = agent.discretize_state(observation)

        total_reward = 0

        for step in range(1000):
            action = agent.choose_action(state)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_observation)

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Test Episode {episode + 1}, Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    # Тренування
    print("Початок тренування...")
    agent, rewards = train_q_learning_opencv(episodes=500, max_steps=1000)

    # Тестування
    print("\nТестування моделі...")
    test_agent()