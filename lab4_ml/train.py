import gymnasium as gym
import numpy as np
import pickle
import cv2
import math
import os
from collections import defaultdict

# --- ВИЗНАЧЕННЯ ДІЙ ДЛЯ CONTINUOUS РЕЖИМУ (Тільки 5 дій) ---
# Ключ: ID дії (для Q-Table)
# Значення: [Steering (-1..1), Gas (0..1), Brake (0..1)]
CONTINUOUS_ACTIONS = {
    0: [0.0, 0.0, 0.0],  # Idle
    1: [0.0, 1.0, 0.0],  # Full Gas (Straight)
    2: [0.0, 0.0, 0.8],  # Brake
    3: [-0.5, 0.2, 0.0],  # Soft Left + Gas (Drift)
    4: [0.5, 0.2, 0.0],  # Soft Right + Gas (Drift)
}

ACTION_NAMES = {
    0: "Idle",
    1: "Gas",
    2: "Brake",
    3: "Left+Gas",
    4: "Right+Gas",
}


class QLearningAgent:
    def __init__(self, action_space_size, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.95, epsilon_min=0.01):
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_space_size = action_space_size

        # --- НАЛАШТУВАННЯ ПРОМЕНІВ (5 штук) ---
        # 0: Far Left, 1: Left, 2: Center, 3: Right, 4: Far Right
        self.ray_angles = [-0.9, -0.4, 0.0, 0.4, 0.9]
        self.ray_length = 35
        self.start_x = 48
        self.start_y = 66

        # --- ДАТЧИК ШВИДКОСТІ (ROI) ---
        self.speed_roi_x = 12
        self.speed_roi_y = 88
        self.speed_roi_w = 3
        self.speed_roi_h = 6

    def cast_rays(self, observation):
        if len(observation.shape) == 3:
            gray = np.mean(observation, axis=2)
        else:
            gray = observation

        distances = []
        endpoints = []

        for angle in self.ray_angles:
            dist = 0
            current_x = self.start_x
            current_y = self.start_y
            step_x = math.sin(angle)
            step_y = -math.cos(angle)
            final_x, final_y = current_x, current_y

            for i in range(self.ray_length):
                current_x += step_x
                current_y += step_y
                if not (0 <= int(current_x) < 96 and 0 <= int(current_y) < 96):
                    break
                pixel_val = gray[int(current_y), int(current_x)]
                if pixel_val > 115:
                    break
                dist += 1
                final_x, final_y = current_x, current_y

            distances.append(dist)
            endpoints.append((int(final_x), int(final_y)))

        return distances, endpoints

    def _get_visual_speed(self, observation):
        x = self.speed_roi_x
        y = self.speed_roi_y
        w = self.speed_roi_w
        h = self.speed_roi_h
        roi = observation[y: y + h, x: x + w, :]
        avg_brightness = np.mean(roi)
        return avg_brightness

    def discretize_state(self, observation):
        raw_distances, _ = self.cast_rays(observation)
        discrete_state = []

        # --- Lidar: 6 рівнів (0-5) ---
        for i in range(len(raw_distances)):
            if raw_distances[i] <= 1:
                discrete_state.append(0)
            elif raw_distances[i] <= 7:
                discrete_state.append(1)
            elif raw_distances[i] <= 13:
                discrete_state.append(2)
            elif raw_distances[i] <= 19:
                discrete_state.append(3)
            elif raw_distances[i] <= 26:
                discrete_state.append(4)
            elif raw_distances[i] <= (self.ray_length - 1):
                discrete_state.append(5)
            else:
                discrete_state.append(6)

        # --- Speed (Visual) ---
        brightness = self._get_visual_speed(observation)
        speed_state = 0
        if brightness < 40:
            speed_state = 0
        elif brightness < 70:
            speed_state = 1
        else:
            speed_state = 2
        discrete_state.append(speed_state)

        return tuple(discrete_state), brightness

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'lr': self.lr,
                'gamma': self.gamma
            }, f)
        print(f"Модель збережено у {filename}")

    def load(self, filename, epsilon=None):
        if not os.path.exists(filename):
            print(f"ПОМИЛКА: Файл '{filename}' не знайдено! Починаємо з нуля.")
            return False

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            loaded_q = data['q_table']
            sample_key = next(iter(loaded_q)) if loaded_q else None

            if sample_key and len(loaded_q[sample_key]) != self.action_space_size:
                print(
                    f"УВАГА: Розмір дій у файлі ({len(loaded_q[sample_key])}) не співпадає з поточним ({self.action_space_size}).")
                print("Неможливо завантажити цю модель. Починаємо з нуля.")
                return False

            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size), data['q_table'])

            if epsilon is not None:
                self.epsilon = epsilon
                print(f"Epsilon перезаписано вручну: {self.epsilon}")
            else:
                self.epsilon = data.get('epsilon', 1.0)

            self.lr = data.get('lr', 0.1)
            self.gamma = data.get('gamma', 0.99)
        print(f"Успішно завантажено модель з '{filename}'. Epsilon: {self.epsilon:.3f}")
        return True


def train_q_learning_opencv(episodes=500,
                            max_steps=1000,
                            load_path=None,
                            save_path="q_learning_car_racing.pkl",
                            start_epsilon=1.0):
    env = gym.make("CarRacing-v3", render_mode="rgb_array",
                   lap_complete_percent=0.95,
                   domain_randomize=False,
                   continuous=True,
                   max_episode_steps=max_steps)

    agent = QLearningAgent(action_space_size=len(CONTINUOUS_ACTIONS))

    if load_path:
        agent.load(load_path, epsilon=start_epsilon)
    elif start_epsilon is not None:
        agent.epsilon = start_epsilon

    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    SCALE_X = WINDOW_WIDTH / 96
    SCALE_Y = WINDOW_HEIGHT / 96

    cv2.namedWindow('Car Racing Continuous', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Car Racing Continuous', WINDOW_WIDTH, WINDOW_HEIGHT)

    episode_rewards = []

    for episode in range(episodes):
        observation, info = env.reset(seed=42)

        state, brightness_val = agent.discretize_state(observation)

        total_reward = 0
        manual_reset = False

        for step in range(max_steps):
            # --- ПРОПУСК ПЕРШИХ 40 КРОКІВ (Zoom animation) ---
            if step < 40:
                next_observation, _, terminated, truncated, _ = env.step(
                    np.array(CONTINUOUS_ACTIONS[0], dtype=np.float32))
                state, brightness_val = agent.discretize_state(next_observation)
                if terminated or truncated: break
                continue
            # --------------------------------

            action_idx = agent.choose_action(state)
            continuous_action = CONTINUOUS_ACTIONS[action_idx]

            next_observation, reward, terminated, truncated, info = env.step(
                np.array(continuous_action, dtype=np.float32))

            next_state, next_brightness_val = agent.discretize_state(next_observation)

            # --- Reward Shaping ---
            if reward < 0:
                reward -= 1
            else:
                reward += 15

            # Розпаковка стану: 5 лідарів + швидкість
            # d1=FarL, d2=L, d3=Center, d4=R, d5=FarR
            d1, d2, d3, d4, d5, speed_class = state
            next_d1, next_d2, next_d3, next_d4, next_d5, speed_class = next_state

            # Логіка для швидкості
            if action_idx == 1:  # Газ
                reward += 2


            if speed_class == 0:
                reward -= 2  # Або стоїмо, або летимо занадто швидко без контролю? (Тут ваша логіка була дивною, але залишаю структуру)
            elif speed_class == 2:
                reward -= 4
            else:
                reward += 2

            # Логіка стін (Бокові: d1, d5. Передні-бокові: d2, d4)
            # Якщо зліва (d1, d2) близько стіна -> караємо
            if d1 <= 2 or d2 <= 2 or d3 <= 2 or d4 <= 2 or d5 <= 2:
                reward -= 3
                # Різкий вліво
                if d1 <= 2 and d2 >= 5 and action_idx == 3:
                    reward += 5
                # Різкий вправо
                elif d5 <= 2 and d4 >= 5 and action_idx == 4:
                    reward += 5

            if d3 < 6:
                if speed_class == 0: reward += 2
                else: reward -= 2
                # Мякий вліво
                if (d2 >= 5 or d1 >= 5) and action_idx == 3: reward += 5
                # Мякий вправо
                elif (d4 >= 5 or d5 >= 5) and action_idx == 4: reward += 5
                # Різкий вліво
                elif d1 >= 3 and d2 <= 2 and d3 <= 2 and d4 <= 2 and d5 <= 2 and action_idx == 3: reward += 5
                # Різкий вправо
                elif d1 <= 2 and d2 <= 2 and d3 <= 2 and d4 <= 2 and d5 >= 3 and action_idx == 3: reward += 5
            else:
                if speed_class == 2: reward -= 5

            #Критична близькість
            if d1 <= 1 or d2 <= 1 or d3 <= 1 or d4 <= 1 or d5 <= 1:
                reward -= 1

            # Краш
            # if d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0 or d5 == 0:
            #     reward -= 5

            # Повний краш (всі сенсори 0)
            if next_d1 == 0 and next_d2 == 0 and next_d3 == 0 and next_d4 == 0 and next_d5 == 0:
                reward -= 20
                agent.update(state, action_idx, reward, next_state, True)
                break

            agent.update(state, action_idx, reward, next_state, terminated or truncated)

            # --- ВІЗУАЛІЗАЦІЯ ---
            frame_render = env.render()
            frame_display = cv2.resize(frame_render, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)
            frame_display = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)

            raw_dists, endpoints = agent.cast_rays(next_observation)
            start_x = int(agent.start_x * SCALE_X)
            start_y = int(agent.start_y * SCALE_Y)

            for i, (end_x, end_y) in enumerate(endpoints):
                dist_class = state[i]
                if dist_class == 6:
                    color = (0, 255, 0)
                elif dist_class == 5:
                    color = (0, 255, 127)
                elif dist_class == 4:
                    color = (0, 255, 255)
                elif dist_class == 3:
                    color = (0, 128, 255)
                elif dist_class == 2:
                    color = (0, 0, 255)
                else:
                    color = (0, 0, 128)

                ex, ey = int(end_x * SCALE_X), int(end_y * SCALE_Y)
                cv2.line(frame_display, (start_x, start_y), (ex, ey), color, 3)
                cv2.circle(frame_display, (ex, ey), 6, color, -1)
                cv2.putText(frame_display, f"{raw_dists[i]}", (ex, ey - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

            hud_y = 30

            def draw_text(text, col=(255, 255, 255)):
                nonlocal hud_y
                cv2.putText(frame_display, text, (12, hud_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(frame_display, text, (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                hud_y += 35

            speed_labels = ['Slow', 'Normal', 'Fast']
            speed_col = (0, 255, 0) if speed_class == 1 else ((0, 255, 255) if speed_class == 0 else (0, 0, 255))
            action_name = ACTION_NAMES[action_idx]

            draw_text(f'Episode: {episode}/{episodes}')
            draw_text(f'Step: {step} / {max_steps}')
            draw_text(f'Action: {action_name} ({action_idx})', (0, 255, 0))
            draw_text(f'Brightness: {brightness_val:.1f} ({speed_labels[speed_class]})', speed_col)
            draw_text(f'Reward: {total_reward:.1f}')
            draw_text(f'Epsilon: {agent.epsilon:.3f}')
            draw_text(f'State: {d1}, {d2}, {d3}, {d4}, {d5}')

            roi_x1 = int(agent.speed_roi_x * SCALE_X)
            roi_y1 = int(agent.speed_roi_y * SCALE_Y)
            roi_x2 = int((agent.speed_roi_x + agent.speed_roi_w) * SCALE_X)
            roi_y2 = int((agent.speed_roi_y + agent.speed_roi_h) * SCALE_Y)
            cv2.rectangle(frame_display, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 255), 2)

            cv2.imshow('Car Racing Continuous', frame_display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                env.close()
                agent.save(save_path)
                return agent, episode_rewards
            elif key == ord('r') or key == ord('R'):
                print(f"Епізод {episode} перервано.")
                manual_reset = True
                break

            state = next_state
            brightness_val = next_brightness_val
            total_reward += reward

            if terminated or truncated:
                break

        if manual_reset: pass

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    cv2.destroyAllWindows()
    env.close()
    agent.save(save_path)
    return agent, episode_rewards


if __name__ == "__main__":
    # Нова назва файлу для 5 лідарів + 5 дій
    model_file = "q_learning_car_racing_continuous_5lidar_5actions.pkl"
    save_file = "q_learning_car_racing_continuous_5lidar_5actions.pkl"

    print("--- CAR RACING Q-LEARNING (5 RAYS + 5 ACTIONS) ---")
    if os.path.exists(model_file):
        print(f"Знайдено {model_file}, продовжуємо...")
        train_q_learning_opencv(episodes=500, max_steps=3000, load_path=model_file, save_path=save_file,
                                start_epsilon=0.02)
    else:
        print("Починаємо з нуля (нова continuous модель)...")
        train_q_learning_opencv(episodes=500, max_steps=3000, save_path=save_file)