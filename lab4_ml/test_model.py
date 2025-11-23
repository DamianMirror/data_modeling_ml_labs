import gymnasium as gym
import numpy as np
import cv2
import os

# Імпортуємо клас агента і налаштування з вашого файлу тренування
# Важливо: Ваш файл тренування має називатися 'train_car_racing_cv2.py'
try:
    from train import QLearningAgent, CONTINUOUS_ACTIONS, ACTION_NAMES
except ImportError:
    print("ПОМИЛКА: Не знайдено файл 'train_car_racing_cv2.py' або в ньому немає потрібних класів.")
    exit()


def run_agent(model_path, episodes=5):
    """
    Запуск агента в режимі тестування (Inference Mode).
    Без навчання, тільки виконання.
    """

    # Створюємо середовище
    env = gym.make("CarRacing-v3", render_mode="rgb_array",
                   lap_complete_percent=0.95,
                   domain_randomize=False,
                   continuous=True)

    # Ініціалізуємо агента
    # Параметри learning_rate і discount_factor тут не важливі, бо ми не вчимося
    agent = QLearningAgent(action_space_size=len(CONTINUOUS_ACTIONS), epsilon=0.0)

    # Завантажуємо ваги
    if os.path.exists(model_path):
        agent.load(model_path, epsilon=0.0)  # Примусово ставимо epsilon 0
        print(f"Модель {model_path} завантажена. Режим: TEST (Epsilon=0).")
    else:
        print(f"Файл {model_path} не знайдено!")
        return

    # Налаштування візуалізації
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    SCALE_X = WINDOW_WIDTH / 96
    SCALE_Y = WINDOW_HEIGHT / 96

    cv2.namedWindow('Car Racing FINAL', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Car Racing FINAL', WINDOW_WIDTH, WINDOW_HEIGHT)

    for episode in range(episodes):
        observation, info = env.reset(seed=42 + episode)  # Різні сіди для різноманіття
        state, brightness_val = agent.discretize_state(observation)

        total_reward = 0
        step = 0

        print(f"Start Episode {episode + 1}...")

        while True:
            step += 1

            # --- ПРОПУСК ПЕРШИХ 40 КРОКІВ (Zoom) ---
            if step < 40:
                action_idx = 0  # Idle
                next_observation, _, terminated, truncated, _ = env.step(
                    np.array(CONTINUOUS_ACTIONS[action_idx], dtype=np.float32))

                state, brightness_val = agent.discretize_state(next_observation)

                # Просто малюємо кадр, щоб не було чорного екрану
                frame_render = env.render()
                frame_display = cv2.resize(frame_render, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)
                frame_display = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_display, "WARMING UP...", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow('Car Racing FINAL', frame_display)
                cv2.waitKey(1)

                if terminated or truncated: break
                continue
            # ---------------------------------------

            # 1. Вибір найкращої дії (Greedy)
            action_idx = agent.choose_action(state)
            continuous_action = CONTINUOUS_ACTIONS[action_idx]

            # 2. Крок
            next_observation, reward, terminated, truncated, info = env.step(
                np.array(continuous_action, dtype=np.float32))

            # 3. Оновлення стану (тільки для вибору наступної дії)
            next_state, next_brightness_val = agent.discretize_state(next_observation)

            # !!! ТУТ НЕМАЄ agent.update(), бо ми не вчимося !!!

            state = next_state
            brightness_val = next_brightness_val
            total_reward += reward

            # --- ВІЗУАЛІЗАЦІЯ ---
            frame_render = env.render()
            frame_display = cv2.resize(frame_render, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)
            frame_display = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)

            # Малюємо лідари
            raw_dists, endpoints = agent.cast_rays(next_observation)
            start_x = int(agent.start_x * SCALE_X)
            start_y = int(agent.start_y * SCALE_Y)

            # Розпаковка стану для кольорів
            # d1, d2, d3, d4, d5, speed
            current_dists_classes = state[:5]

            for i, (end_x, end_y) in enumerate(endpoints):
                dist_class = current_dists_classes[i]

                # Кольори (ті самі, що в тренуванні)
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
                cv2.line(frame_display, (start_x, start_y), (ex, ey), color, 2)
                cv2.circle(frame_display, (ex, ey), 5, color, -1)

            # HUD
            hud_y = 30

            def draw_text(text, col=(255, 255, 255)):
                nonlocal hud_y
                cv2.putText(frame_display, text, (12, hud_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(frame_display, text, (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                hud_y += 35

            speed_labels = ['Slow', 'Normal', 'Fast']
            current_speed_class = state[5]
            speed_col = (0, 255, 0) if current_speed_class == 1 else (
                (0, 255, 255) if current_speed_class == 0 else (0, 0, 255))
            action_name = ACTION_NAMES[action_idx]

            draw_text(f'TEST MODE | Episode: {episode + 1}', (0, 255, 255))
            draw_text(f'Action: {action_name}', (0, 255, 0))
            draw_text(f'Brightness: {brightness_val:.1f} ({speed_labels[current_speed_class]})', speed_col)
            draw_text(f'Total Reward: {total_reward:.1f}')
            draw_text(f'Lidar State: {current_dists_classes}')

            # ROI Rectangle
            roi_x1 = int(agent.speed_roi_x * SCALE_X)
            roi_y1 = int(agent.speed_roi_y * SCALE_Y)
            roi_x2 = int((agent.speed_roi_x + agent.speed_roi_w) * SCALE_X)
            roi_y2 = int((agent.speed_roi_y + agent.speed_roi_h) * SCALE_Y)
            cv2.rectangle(frame_display, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 255), 2)

            # Інструкція
            cv2.putText(frame_display, "Press 'R' to reset, 'ESC' to quit", (WINDOW_WIDTH - 450, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow('Car Racing FINAL', frame_display)

            # Обробка клавіш
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                env.close()
                return
            elif key == ord('r') or key == ord('R'):
                print("Restarting episode...")
                break

            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}")
                break

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    # Вкажіть тут файл вашої фінальної моделі
    MODEL_FILE = "q_learning_car_racing_continuous_5lidar_5actions.pkl"

    print(f"Запуск тестування моделі: {MODEL_FILE}")
    run_agent(MODEL_FILE, episodes=10)