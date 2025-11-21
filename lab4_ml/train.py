import gymnasium as gym
import numpy as np
import pickle
import cv2
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

        # --- ЄДИНЕ МІСЦЕ НАЛАШТУВАННЯ ROI ---
        # Визначаємо, куди дивиться агент
        self.roi_top = 40
        self.roi_bottom = 65
        self.roi_left = 35
        self.roi_right = 61

    def discretize_state(self, observation):
        """Спрощення стану до дискретного представлення, використовуючи збережені ROI"""

        # Використовуємо self.roi_... замість хардкоду
        roi = observation[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]

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
                'gamma': self.gamma,
                # Можна також зберегти конфігурацію ROI, щоб знати, як була навчена модель
                'roi_config': (self.roi_top, self.roi_bottom, self.roi_left, self.roi_right)
            }, f)
        print(f"Модель збережено у {filename}")

    def load(self, filename):
        """Завантажити модель"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size), data['q_table'])
            self.epsilon = data.get('epsilon', 1.0)
            self.lr = data.get('lr', 0.1)
            self.gamma = data.get('gamma', 0.99)

            # Якщо в файлі є конфіг ROI, можна його завантажити або попередити користувача
            if 'roi_config' in data:
                saved_roi = data['roi_config']
                current_roi = (self.roi_top, self.roi_bottom, self.roi_left, self.roi_right)
                if saved_roi != current_roi:
                    print(f"УВАГА: ROI моделі {saved_roi} відрізняється від поточного {current_roi}!")

        print(f"Модель завантажено з {filename}")


def train_q_learning_opencv(episodes=500, max_steps=1000, save_path="q_learning_car_racing.pkl"):
    """
    Тренування з візуалізацією через OpenCV.
    Використовує ROI безпосередньо з налаштувань агента.
    """

    # Ініціалізація середовища
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95,
                   domain_randomize=False, continuous=False)

    agent = QLearningAgent(action_space_size=5)
    episode_rewards = []

    # Налаштування візуалізації
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    GAME_WIDTH = 96  # Оригінальна ширина Gym
    GAME_HEIGHT = 96  # Оригінальна висота Gym

    # Коефіцієнти масштабування
    SCALE_X = WINDOW_WIDTH / GAME_WIDTH
    SCALE_Y = WINDOW_HEIGHT / GAME_HEIGHT

    cv2.namedWindow('Car Racing Training', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Car Racing Training', WINDOW_WIDTH, WINDOW_HEIGHT)

    for episode in range(episodes):
        observation, info = env.reset(seed=42)
        state = agent.discretize_state(observation)
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_observation)

            # --- Логіка нагороди (Reward Shaping) ---
            # if reward < 0:
            #     reward = -10
            # else:
            #     reward += 0.5

            if action == 3:  # Газ
                reward += 0.5
            elif action == 0:  # Нічого
                reward -= 100
            elif action == 4:  # Гальмо
                reward -= 0.1

            left_class, center_class, right_class = state

            # Ідеальна позиція: центр = дорога (2), краї = край/трава (0 або 1)
            if center_class == 2 and (left_class <= 1) and (right_class <= 1):
                reward += 5  # Бонус за їзду по центру дороги
            elif left_class <= 1 and (center_class == 2) and (right_class == 2):
                if action == 1: reward += 5
                reward -= 5
            elif left_class == 2 and (center_class == 2) and (left_class <= 1):
                reward -= 5
            elif center_class == 2:
                reward += 2  # Невеликий бонус за те, що хоча б центр на дорозі

            agent.update(state, action, reward, next_state, terminated or truncated)

            # --- ВІЗУАЛІЗАЦІЯ ---
            frame = env.render()  # Отримуємо оригінал 96x96

            # 1. Resize
            frame_display = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)
            frame_display = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)

            # 2. Беремо координати ПРЯМО З АГЕНТА (Single Source of Truth)
            roi_top_scaled = int(agent.roi_top * SCALE_Y)
            roi_bottom_scaled = int(agent.roi_bottom * SCALE_Y)
            roi_left_scaled = int(agent.roi_left * SCALE_X)
            roi_right_scaled = int(agent.roi_right * SCALE_X)

            # Обчислення зон всередині ROI
            roi_width_scaled = roi_right_scaled - roi_left_scaled
            zone1_x = roi_left_scaled + (roi_width_scaled // 3)
            zone2_x = roi_left_scaled + (2 * roi_width_scaled // 3)

            # Отримання інформації про стан
            left_class, center_class, right_class = state

            def get_class_info(cls):
                if cls == 2:
                    return "Road", (100, 100, 100)  # Сірий
                elif cls == 1:
                    return "Edge", (0, 165, 255)  # Помаранчевий
                else:
                    return "Grass", (0, 255, 0)  # Зелений

            _, c_left = get_class_info(left_class)
            _, c_center = get_class_info(center_class)
            _, c_right = get_class_info(right_class)

            # 4. Малюємо напівпрозору заливку
            overlay = frame_display.copy()
            cv2.rectangle(overlay, (roi_left_scaled, roi_top_scaled), (zone1_x, roi_bottom_scaled), c_left, -1)
            cv2.rectangle(overlay, (zone1_x, roi_top_scaled), (zone2_x, roi_bottom_scaled), c_center, -1)
            cv2.rectangle(overlay, (zone2_x, roi_top_scaled), (roi_right_scaled, roi_bottom_scaled), c_right, -1)

            cv2.addWeighted(overlay, 0.3, frame_display, 0.7, 0, frame_display)

            # 5. Малюємо рамки
            cv2.rectangle(frame_display, (roi_left_scaled, roi_top_scaled), (roi_right_scaled, roi_bottom_scaled),
                          (0, 0, 255), 2)
            cv2.line(frame_display, (zone1_x, roi_top_scaled), (zone1_x, roi_bottom_scaled), (0, 255, 255), 2)
            cv2.line(frame_display, (zone2_x, roi_top_scaled), (zone2_x, roi_bottom_scaled), (0, 255, 255), 2)

            # 6. Текст
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            cv2.putText(frame_display, str(left_class), (roi_left_scaled + 5, roi_top_scaled - 5), font, font_scale,
                        c_left, thickness)
            cv2.putText(frame_display, str(center_class), (zone1_x + 5, roi_top_scaled - 5), font, font_scale, c_center,
                        thickness)
            cv2.putText(frame_display, str(right_class), (zone2_x + 5, roi_top_scaled - 5), font, font_scale, c_right,
                        thickness)

            # 7. HUD
            action_names = ['Nothing', 'Left', 'Right', 'Gas', 'Brake']
            hud_x = 10
            hud_y = 30
            line_h = 35

            def draw_hud(text, color=(255, 255, 255)):
                nonlocal hud_y
                cv2.putText(frame_display, text, (hud_x, hud_y), font, 0.7, (0, 0, 0), 4)
                cv2.putText(frame_display, text, (hud_x, hud_y), font, 0.7, color, 2)
                hud_y += line_h

            draw_hud(f'Episode: {episode}/{episodes}')
            draw_hud(f'Step: {step}')
            draw_hud(f'Action: {action_names[action]}', (0, 255, 0))
            draw_hud(f'Reward: {total_reward:.1f}', (255, 255, 0))
            draw_hud(f'Epsilon: {agent.epsilon:.3f}', (255, 0, 255))

            cv2.imshow('Car Racing Training', frame_display)

            if cv2.waitKey(1) & 0xFF == 27:
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
    agent.epsilon = 0

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