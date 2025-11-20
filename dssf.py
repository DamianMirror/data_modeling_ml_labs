import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# =============================
# 1. Завантаження даних
# =============================

data = pd.read_csv("Housing.csv")

# Обираємо 2 числові змінні для аналізу
x1 = data["price"].dropna()
x2 = data["area"].dropna()

# =============================
# 2. Функція для перевірки нормальності
# =============================

def check_normality(data, var_name):
    print(f"\n=== Перевірка для змінної {var_name} ===")

    # гістограма
    plt.figure(figsize=(10, 4))
    plt.hist(data, bins=30, density=True, alpha=0.6, color="blue")
    plt.title(f"Гістограма {var_name}")
    plt.xlabel(var_name)
    plt.ylabel("Частота")
    plt.show()

    # Q-Q діаграма
    plt.figure(figsize=(6, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Q-Q графік для {var_name}")
    plt.show()

    # Тест Колмогорова-Смирнова
    standardized = (data - np.mean(data)) / np.std(data, ddof=1)
    stat, p = stats.kstest(standardized, 'norm')
    print(f"Тест Колмогорова-Смирнова: статистика={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print("Дані можна вважати нормально розподіленими")
    else:
        print("Дані не є нормально розподіленими")

# Перевірка початкових даних
check_normality(x1, "price")
check_normality(x2, "area")

# =============================
# 3. Функція для трансформацій і повторної перевірки
# =============================

def transform_and_check(data, var_name):
    print(f"\n--- Трансформації для {var_name} ---")

    # Логарифм
    log_data = np.log1p(data)  # log(1+x), щоб уникнути проблем з нулями
    check_normality(log_data, var_name + " (log)")

    # Квадратний корінь
    sqrt_data = np.sqrt(data)
    check_normality(sqrt_data, var_name + " (sqrt)")

    # Кубічний корінь
    cbrt_data = np.cbrt(data)
    check_normality(cbrt_data, var_name + " (cbrt)")

# Трансформації для price і area
transform_and_check(x1, "price")
transform_and_check(x2, "area")