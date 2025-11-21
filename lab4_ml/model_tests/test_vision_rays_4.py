import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math


def test_lidar_visualization():
    # 1. Налаштування середовища
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    observation, info = env.reset(seed=42)

    print("Симуляція 50 кроків...")
    # 2. Проганяємо 45 кроків (щоб набрати трохи швидкості і виїхати на пряму)
    for i in range(45):
        action = 3  # Gas
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset(seed=42)

    print("Крок 50. Починаємо аналіз променів...")

    # 3. Отримуємо дані для аналізу
    # Конвертуємо в grayscale (середнє по каналах)
    gray_frame = np.mean(observation, axis=2)

    # Параметри променів (ОНОВЛЕНО: 4 ПРОМЕНІ)
    start_x, start_y = 48, 66  # Центр капота
    ray_length = 40

    # Нові кути, як в агенті
    ray_angles = [-0.6, -0.2, 0.2, 0.6]

    ray_labels = ['Far Left (-0.6)', 'Left (-0.2)', 'Right (0.2)', 'Far Right (0.6)']
    # Кольори для 4 променів (червоний, помаранчевий, блакитний, синій)
    ray_colors = ['red', 'orange', 'cyan', 'blue']

    # Зберігатимемо дані для графіків тут
    rays_visual_data = []  # (x_path, y_path) для малювання ліній на картинці
    rays_intensity_data = []  # Значення сірого на кожному кроці променя

    threshold_val = 115  # Поріг трави

    # --- ЛОГІКА RAY CASTING ---
    for angle in ray_angles:
        current_x, current_y = start_x, start_y

        # Обчислюємо крок (вектор напрямку)
        step_x = math.sin(angle)
        step_y = -math.cos(angle)

        # Списки для збереження шляху одного променя
        path_x = [current_x]
        path_y = [current_y]
        intensities = []  # Яскравість пікселів

        hit = False

        for i in range(ray_length):
            # Крокуємо
            current_x += step_x
            current_y += step_y

            # Перевірка меж
            if not (0 <= int(current_x) < 96 and 0 <= int(current_y) < 96):
                break

            # Отримуємо значення пікселя (0..255)
            px_value = gray_frame[int(current_y), int(current_x)]

            path_x.append(current_x)
            path_y.append(current_y)
            intensities.append(px_value)

            # Перевірка на зіткнення з травою
            if px_value > threshold_val:
                hit = True
                # Ми не перериваємо цикл break-ом тут,
                # щоб на графіку було видно, як виглядає "трава" далі
                # але зупиняємось трохи після удару для наочності
                if len(intensities) > 5 and intensities[-2] > threshold_val:
                    break

        rays_visual_data.append((path_x, path_y))
        rays_intensity_data.append(intensities)

    env.close()

    # 4. ВІЗУАЛІЗАЦІЯ MATPLOTLIB
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Графік 1: Зображення з променями ---
    ax1.set_title("Frame 50 + 4 Lidar Rays")
    ax1.imshow(observation)  # Показуємо оригінальне кольорове зображення

    # Малюємо точку старту
    ax1.scatter([start_x], [start_y], c='yellow', s=50, marker='o', label='Start')

    for i, (path_x, path_y) in enumerate(rays_visual_data):
        ax1.plot(path_x, path_y, color=ray_colors[i], linewidth=2, label=ray_labels[i])
        # Малюємо точку кінця
        ax1.scatter([path_x[-1]], [path_y[-1]], c=ray_colors[i], s=30)

    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 96)
    ax1.set_ylim(96, 0)  # Інверсія Y

    # --- Графік 2: Значення пікселів вздовж променя ---
    ax2.set_title("Grayscale Intensity along Rays")
    ax2.set_xlabel("Distance (pixels)")
    ax2.set_ylabel("Pixel Intensity (0-255)")

    # Малюємо поріг
    ax2.axhline(y=threshold_val, color='black', linestyle='--', alpha=0.5, label=f'Grass Threshold ({threshold_val})')

    for i, intensities in enumerate(rays_intensity_data):
        distances = range(len(intensities))
        ax2.plot(distances, intensities, color=ray_colors[i], marker='.', label=ray_labels[i])

        # Знаходимо де саме стався "удар" (перше перевищення порогу)
        hit_idx = next((idx for idx, val in enumerate(intensities) if val > threshold_val), None)
        if hit_idx is not None:
            ax2.scatter([hit_idx], [intensities[hit_idx]], c='black', s=100, marker='x', zorder=10)
            # ax2.text(hit_idx, intensities[hit_idx] + 5, f"Hit {hit_idx}", fontsize=8)

    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_lidar_visualization()