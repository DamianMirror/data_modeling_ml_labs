import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- НАЛАШТУВАННЯ ЗОНИ ПОШУКУ (ROI) ---
# Ваші нові координати
SPEED_ROI_X = 12
SPEED_ROI_Y = 88
SPEED_ROI_WIDTH = 3
SPEED_ROI_HEIGHT = 6


def test_speed_logic():
    # 1. Ініціалізація
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    observation, info = env.reset(seed=42)

    snapshots = []

    print("Старт симуляції на 50 кроків (тільки ГАЗ)...")

    # 2. Цикл симуляції
    for step in range(51):
        # Тиснемо тільки газ (дія 3)
        observation, reward, terminated, truncated, info = env.step(3)

        if step % 10 == 0:
            # --- ЛОГІКА (Використовуємо константи зверху) ---
            x1 = SPEED_ROI_X
            x2 = SPEED_ROI_X + SPEED_ROI_WIDTH
            y1 = SPEED_ROI_Y
            y2 = SPEED_ROI_Y + SPEED_ROI_HEIGHT

            # Вирізаємо маленьку зону
            speed_roi = observation[y1:y2, x1:x2, :]

            # НОВА ЛОГІКА: Середня яскравість
            # Рахуємо середнє арифметичне всіх каналів
            avg_brightness = np.mean(speed_roi)

            snapshots.append({
                'step': step,
                'full_frame': observation.copy(),
                'roi': speed_roi,
                'brightness': avg_brightness,
                'coords': (x1, y1, SPEED_ROI_WIDTH, SPEED_ROI_HEIGHT)
            })

            print(f"Крок {step}: середня яскравість зони = {avg_brightness:.2f}")

        if terminated or truncated:
            observation, info = env.reset(seed=42)

    env.close()

    # 3. Візуалізація результатів
    num_snaps = len(snapshots)
    fig, axes = plt.subplots(3, num_snaps, figsize=(15, 6))

    # Підписи рядків
    axes[0, 0].set_ylabel("Full Frame\n(Green Box = ROI)", fontsize=12)
    axes[1, 0].set_ylabel(f"Zoomed ROI\n({SPEED_ROI_WIDTH}x{SPEED_ROI_HEIGHT})", fontsize=12)
    axes[2, 0].set_ylabel("Brightness Perception\n(Grayscale)", fontsize=12)

    for i, snap in enumerate(snapshots):
        # 1. Повний кадр
        ax_full = axes[0, i]
        ax_full.imshow(snap['full_frame'])

        # Малюємо прямокутник зони пошуку
        (x, y, w, h) = snap['coords']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='#00FF00', facecolor='none')
        ax_full.add_patch(rect)

        ax_full.set_title(f"Step {snap['step']}")
        ax_full.axis('off')

        # 2. Сама зона (збільшено, кольорова)
        ax_roi = axes[1, i]
        ax_roi.imshow(snap['roi'])

        # Рамка навколо зони
        for spine in ax_roi.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2)
        ax_roi.set_xticks([])
        ax_roi.set_yticks([])

        # 3. Як це бачить алгоритм (чорно-біле / яскравість)
        ax_val = axes[2, i]
        # Конвертуємо ROI в grayscale для візуалізації того, що ми усереднюємо
        roi_gray = np.mean(snap['roi'], axis=2)
        ax_val.imshow(roi_gray, cmap='gray', vmin=0, vmax=255)

        # Виводимо числове значення
        val = snap['brightness']
        color = 'red' if val < 10 else ('orange' if val < 50 else 'green')
        ax_val.set_title(f"Value: {val:.1f}", color=color, fontweight='bold', fontsize=14)
        ax_val.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_speed_logic()