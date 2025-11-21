import gymnasium as gym
import numpy as np
import pickle
import cv2
import math
import os
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

        # --- НАЛАШТУВАННЯ ПРОМЕНІВ (LIDAR - 4 PROBES) ---
        # 4 промені: [Side Left, Front Left, Front Right, Side Right]
        self.ray_angles = [-0.6, -0.2, 0.2, 0.6]
        self.ray_length = 40  # Дальність огляду в пікселях
        self.start_x = 48
        self.start_y = 66

        # --- НАЛАШТУВАННЯ ДАТЧИКА ШВИДКОСТІ (ROI) ---
        # Повернули ваші параметри
        self.speed_roi_x = 12
        self.speed_roi_y = 88
        self.speed_roi_w = 3
        self.speed_roi_h = 6

    def cast_rays(self, observation):
        """Пускає промені і повертає точні відстані."""
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
                if pixel_val > 115:  # Трава
                    break
                dist += 1
                final_x, final_y = current_x, current_y

            distances.append(dist)
            endpoints.append((int(final_x), int(final_y)))

        return distances, endpoints

    def _get_visual_speed(self, observation):
        """
        Старий метод: рахуємо середню яскравість у зоні ROI.
        """
        x = self.speed_roi_x
        y = self.speed_roi_y
        w = self.speed_roi_w
        h = self.speed_roi_h
        roi = observation[y: y + h, x: x + w, :]
        avg_brightness = np.mean(roi)
        return avg_brightness

    def discretize_state(self, observation, steering_angle):
        """
        Перетворює стан у дискретний кортеж.
        Швидкість береться ВІЗУАЛЬНО, Кермо - з фізики.
        """
        raw_distances, _ = self.cast_rays(observation)
        discrete_state = []

        # 1. Lidar (4 промені)
        for i, d in enumerate(raw_distances):
            is_side_sensor = (i == 0 or i == 3)

            if is_side_sensor:
                # БОКОВІ ЛІДАРИ
                if d <= 2:
                    discrete_state.append(0)
                elif d <= 6:
                    discrete_state.append(1)
                elif d <= 12:
                    discrete_state.append(2)
                elif d <= 16:
                    discrete_state.append(3)
                else:
                    discrete_state.append(4)
            else:
                # ПЕРЕДНІ ЛІДАРИ
                if d <= 3:
                    discrete_state.append(0)
                elif d <= 8:
                    discrete_state.append(1)
                elif d <= 15:
                    discrete_state.append(2)
                elif d <= 30:
                    discrete_state.append(3)
                else:
                    discrete_state.append(4)

        # 2. Speed (VISUAL)
        brightness = self._get_visual_speed(observation)
        speed_state = 0
        if brightness < 40:
            speed_state = 0  # Slow
        elif brightness < 60:
            speed_state = 1  # Normal
        else:
            speed_state = 2  # Fast
        discrete_state.append(speed_state)

        # 3. Steering (PHYSICS)
        steering_state = 1  # Straight
        if steering_angle < -0.1:
            steering_state = 0  # Left
        elif steering_angle > 0.1:
            steering_state = 2  # Right

        discrete_state.append(steering_state)

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
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size), data['q_table'])

            if epsilon is not None:
                self.epsilon = epsilon
                print(f"Epsilon перезаписано вручну: {self.epsilon}")
            else:
                self.epsilon = data.get('epsilon', 1.0)

            self.lr = data.get('lr', 0.1)
            self.gamma = data.get('gamma', 0.99)
        print(f"Успішно завантажено модель з '{filename}'. Продовжуємо навчання з Epsilon: {self.epsilon:.3f}")
        return True


def train_q_learning_opencv(episodes=500,
                            max_steps=1000,
                            load_path=None,
                            save_path="q_learning_car_racing.pkl",
                            start_epsilon=1.0):
    env = gym.make("CarRacing-v3", render_mode="rgb_array",
                   lap_complete_percent=0.95,
                   domain_randomize=False,
                   continuous=False,
                   max_episode_steps=max_steps)

    agent = QLearningAgent(action_space_size=5)

    if load_path:
        agent.load(load_path, epsilon=start_epsilon)
    elif start_epsilon is not None:
        agent.epsilon = start_epsilon

    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    SCALE_X = WINDOW_WIDTH / 96
    SCALE_Y = WINDOW_HEIGHT / 96

    cv2.namedWindow('Car Racing Lidar', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Car Racing Lidar', WINDOW_WIDTH, WINDOW_HEIGHT)

    episode_rewards = []

    for episode in range(episodes):
        observation, info = env.reset(seed=42)

        # Отримуємо кермо для початкового стану
        try:
            steering_angle = env.unwrapped.car.wheels[0].joint.angle
        except AttributeError:
            steering_angle = env.unwrapped.car.wheels[0].joints[0].joint.angle

        state, brightness_val = agent.discretize_state(observation, steering_angle)

        total_reward = 0
        manual_reset = False

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_observation, reward, terminated, truncated, info = env.step(action)

            # --- Отримання керма ---
            try:
                next_steering_angle = env.unwrapped.car.wheels[0].joint.angle
            except AttributeError:
                next_steering_angle = env.unwrapped.car.wheels[0].joints[0].joint.angle

            # discretize_state сама порахує швидкість з картинки
            next_state, next_brightness_val = agent.discretize_state(next_observation, next_steering_angle)

            # --- Reward Shaping ---
            if reward < 0:
                reward -= 1
            else:
                reward += 10

            if action == 3:
                reward += 2  # Газ
            elif action == 0:
                reward -= 0.5
            elif action == 4:
                reward -= 0.5

            d1, d2, d3, d4, speed_class, steer_class = state

            # Логіка швидкості (тепер спирається на Visual Speed)
            if speed_class == 0 or speed_class == 2:
                reward -= 5
            elif speed_class == 1:
                reward += 2

            if d1 <= 2 and action == 1:
                reward += 2
            elif d4 <= 2 and action == 2:
                reward += 2

            if abs(d1 - d4) >= 1: reward -= 3
            if abs(d2 - d3) >= 1: reward -= 1

            if d2 >= 4 and d3 >= 4:
                if speed_class == 2: reward += 2

            if d1 <= 1 or d2 <= 1 or d3 <= 1 or d4 <= 1: reward -= 5

            if d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0: reward -= 20

            if d1 == 0 and d2 == 0 and d3 == 0 and d4 == 0:
                reward -= 50
                agent.update(state, action, reward, next_state, True)
                break

            agent.update(state, action, reward, next_state, terminated or truncated)

            # --- ВІЗУАЛІЗАЦІЯ ---
            frame_render = env.render()
            frame_display = cv2.resize(frame_render, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)
            frame_display = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)

            # Лідар
            raw_dists, endpoints = agent.cast_rays(next_observation)
            start_x = int(agent.start_x * SCALE_X)
            start_y = int(agent.start_y * SCALE_Y)

            for i, (end_x, end_y) in enumerate(endpoints):
                dist_class = state[i]

                if dist_class == 4:
                    color = (0, 255, 0)
                elif dist_class == 3:
                    color = (0, 255, 128)
                elif dist_class == 2:
                    color = (0, 255, 255)
                elif dist_class == 1:
                    color = (0, 128, 255)
                else:
                    color = (0, 0, 255)

                ex, ey = int(end_x * SCALE_X), int(end_y * SCALE_Y)
                cv2.line(frame_display, (start_x, start_y), (ex, ey), color, 3)
                cv2.circle(frame_display, (ex, ey), 6, color, -1)
                cv2.putText(frame_display, f"{raw_dists[i]}", (ex, ey - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

            # HUD
            hud_y = 30

            def draw_text(text, col=(255, 255, 255)):
                nonlocal hud_y
                cv2.putText(frame_display, text, (12, hud_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(frame_display, text, (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                hud_y += 35

            speed_labels = ['Slow', 'Normal', 'Fast']
            steer_labels = ['Left', 'Straight', 'Right']
            speed_col = (0, 255, 0) if speed_class == 1 else ((0, 255, 255) if speed_class == 0 else (0, 0, 255))

            draw_text(f'Episode: {episode}/{episodes}')
            draw_text(f'Step: {step}/{max_steps}')
            draw_text(f'Action: {action}', (0, 255, 0))

            # Показуємо візуальну яскравість
            draw_text(f'Brightness: {brightness_val:.1f} ({speed_labels[speed_class]})', speed_col)

            st_col = (0, 255, 255) if steer_class == 1 else (0, 100, 255)
            draw_text(f'Steering: {steering_angle:.2f} ({steer_labels[steer_class]})', st_col)

            draw_text(f'Reward: {total_reward:.1f}')
            draw_text(f'Epsilon: {agent.epsilon:.3f}')
            draw_text(f'Lidar: {d1}-{d2}-{d3}-{d4}')

            # ROI Rectangle (Відображаємо зону пошуку швидкості)
            roi_x1 = int(agent.speed_roi_x * SCALE_X)
            roi_y1 = int(agent.speed_roi_y * SCALE_Y)
            roi_x2 = int((agent.speed_roi_x + agent.speed_roi_w) * SCALE_X)
            roi_y2 = int((agent.speed_roi_y + agent.speed_roi_h) * SCALE_Y)
            cv2.rectangle(frame_display, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 255), 2)

            # Інструкція
            cv2.putText(frame_display, "Press 'R' to reset", (WINDOW_WIDTH - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 165, 255), 2)

            cv2.imshow('Car Racing Lidar', frame_display)

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
    model_file = "q_learning_car_racing_4_lidar_steer.pkl"
    save_file = "q_learning_car_racing_4_lidar_steer.pkl"

    print("--- CAR RACING Q-LEARNING (VISUAL SPEED + STEER) ---")
    if os.path.exists(model_file):
        print(f"Знайдено {model_file}, продовжуємо...")
        train_q_learning_opencv(episodes=500, max_steps=500, load_path=model_file, save_path=save_file,
                                start_epsilon=0.01)
    else:
        print("Починаємо з нуля...")
        train_q_learning_opencv(episodes=500, max_steps=3000, save_path=save_file)