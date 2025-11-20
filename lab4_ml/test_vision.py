import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_roi_zones(step_number=50):
    """Візуалізація ROI та зон для налаштування дискретизації"""

    # Створюємо середовище
    env = gym.make("CarRacing-v3", render_mode="rgb_array",
                   lap_complete_percent=0.95, domain_randomize=False, continuous=False)

    observation, info = env.reset(seed=42)

    # Робимо step_number кроків
    print(f"🏎️ Пропускаємо початкову анімацію ({step_number} кроків)...")
    for step in range(step_number):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if step % 10 == 0:
            print(f"   Крок {step}...")

    print(f"✅ Зупинились на кроці {step_number}")
    print(f"   Остання дія: {action}, винагорода: {reward:.2f}")

    env.close()

    # НАЛАШТУВАННЯ ROI - ТУТ МОЖНА ЗМІНЮВАТИ!
    roi_top = 40
    roi_bottom = 65
    roi_left = 25
    roi_right = 71

    # Вирізаємо ROI
    roi = observation[roi_top:roi_bottom, roi_left:roi_right]

    # Перетворюємо в grayscale
    gray = np.mean(roi, axis=2)

    # Ділимо на 3 зони
    width = roi.shape[1]
    zone1_end = width // 3
    zone2_end = 2 * width // 3

    left_zone = gray[:, :zone1_end]
    center_zone = gray[:, zone1_end:zone2_end]
    right_zone = gray[:, zone2_end:]

    # Обчислюємо середні значення
    left_val = np.mean(left_zone)
    center_val = np.mean(center_zone)
    right_val = np.mean(right_zone)

    print(f"\n📊 Середні значення яскравості:")
    print(f"   Ліва зона:       {left_val:.2f}")
    print(f"   Центральна зона: {center_val:.2f}")
    print(f"   Права зона:      {right_val:.2f}")

    # Створюємо візуалізацію
    fig = plt.figure(figsize=(16, 10))

    # 1. Повне зображення з рамкою ROI
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(observation)
    ax1.set_title('Повне зображення', fontsize=14, fontweight='bold')
    rect = patches.Rectangle((roi_left, roi_top),
                             roi_right - roi_left,
                             roi_bottom - roi_top,
                             linewidth=3, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.axhline(y=roi_top, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=roi_bottom, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=roi_left, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=roi_right, color='red', linestyle='--', alpha=0.5)
    ax1.text(roi_left, roi_top - 5, 'ROI', color='red', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.axis('off')

    # 2. ROI (Region of Interest)
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(roi)
    ax2.set_title('ROI (вирізана область)', fontsize=14, fontweight='bold')
    # Лінії розділення зон
    ax2.axvline(x=zone1_end, color='yellow', linewidth=2, linestyle='--')
    ax2.axvline(x=zone2_end, color='yellow', linewidth=2, linestyle='--')
    ax2.text(zone1_end / 2, 5, 'LEFT', color='yellow', fontsize=10,
             fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax2.text(zone1_end + (zone2_end - zone1_end) / 2, 5, 'CENTER', color='yellow',
             fontsize=10, fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax2.text(zone2_end + (width - zone2_end) / 2, 5, 'RIGHT', color='yellow',
             fontsize=10, fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax2.axis('off')

    # 3. Grayscale ROI
    ax3 = plt.subplot(2, 3, 3)
    im = ax3.imshow(gray, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('Grayscale + пороги', fontsize=14, fontweight='bold')
    ax3.axvline(x=zone1_end, color='yellow', linewidth=2, linestyle='--')
    ax3.axvline(x=zone2_end, color='yellow', linewidth=2, linestyle='--')
    plt.colorbar(im, ax=ax3, label='Яскравість (0-255)')
    ax3.axis('off')

    # 4. Ліва зона
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(left_zone, cmap='gray', vmin=0, vmax=255)
    ax4.set_title(f'Ліва зона\nСереднє: {left_val:.2f}',
                  fontsize=12, fontweight='bold', color='blue')
    plt.colorbar(im4, ax=ax4)
    ax4.axis('off')

    # 5. Центральна зона
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(center_zone, cmap='gray', vmin=0, vmax=255)
    ax5.set_title(f'Центральна зона\nСереднє: {center_val:.2f}',
                  fontsize=12, fontweight='bold', color='green')
    plt.colorbar(im5, ax=ax5)
    ax5.axis('off')

    # 6. Права зона
    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(right_zone, cmap='gray', vmin=0, vmax=255)
    ax6.set_title(f'Права зона\nСереднє: {right_val:.2f}',
                  fontsize=12, fontweight='bold', color='red')
    plt.colorbar(im6, ax=ax6)
    ax6.axis('off')

    plt.tight_layout()
    plt.suptitle(f'Візуалізація зон дискретизації (крок {step_number})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.show()

    # Виводимо рекомендації по порогам
    print(f"\n💡 Рекомендовані пороги для classify():")
    all_vals = [left_val, center_val, right_val]
    min_val = min(all_vals)
    max_val = max(all_vals)

    threshold1 = min_val + (max_val - min_val) * 0.33
    threshold2 = min_val + (max_val - min_val) * 0.66

    print(f"   if val < {threshold1:.1f}:")
    print(f"       return 0  # темна зона (трава/поза дорогою)")
    print(f"   elif val < {threshold2:.1f}:")
    print(f"       return 1  # середня зона (край дороги)")
    print(f"   else:")
    print(f"       return 2  # світла зона (дорога)")

    print(f"\n⚙️ Поточні координати ROI:")
    print(f"   roi_top    = {roi_top}")
    print(f"   roi_bottom = {roi_bottom}")
    print(f"   roi_left   = {roi_left}")
    print(f"   roi_right  = {roi_right}")
    print(f"\n📝 Змінюй ці значення в коді, щоб підлаштувати ROI!")

    # Додаткова інформація про розподіл значень
    print(f"\n📈 Статистика яскравості ROI:")
    print(f"   Min: {gray.min():.2f}")
    print(f"   Max: {gray.max():.2f}")
    print(f"   Mean: {gray.mean():.2f}")
    print(f"   Std: {gray.std():.2f}")


if __name__ == "__main__":
    print("🚗 Запуск візуалізації зон для CarRacing...")
    print("=" * 60)

    # Можна змінити номер кроку тут
    visualize_roi_zones(step_number=50)

    print("=" * 60)
    print("✅ Готово! Подивись на графіки і підлаштуй координати ROI.")
    print("\n💡 Підказка: якщо хочеш подивитись на інший момент гри,")
    print("   змінюй параметр step_number (наприклад, 100, 200, тощо)")