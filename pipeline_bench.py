import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from Models import WDTW, WDTWClassifier

# Библиотеки для baseline и данных
from sktime.datasets import load_UCR_UEA_dataset
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import precision_score, accuracy_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


print("=== НАЧАЛО ВЫПОЛНЕНИЯ ПАЙПЛАЙНА ===")
print("Шаг 1: Загрузка датасета BasicMotions из архива UCR/UEA...")
X_train_raw, y_train_raw = load_UCR_UEA_dataset(name="BasicMotions", split="train", return_type="numpy3d")
X_test_raw, y_test_raw = load_UCR_UEA_dataset(name="BasicMotions", split="test", return_type="numpy3d")
X_train_raw = X_train_raw[:, 0:1, :]
X_test_raw = X_test_raw[:, 0:1, :]

num_samples, num_channels, seq_len = X_train_raw.shape
unique_labels = np.unique(y_train_raw)
print(f"Успешно. Сигналов: {num_samples}, Длина: {seq_len}, Каналов: {num_channels}, Классов: {len(unique_labels)}")

# Форматирование под PyTorch
X_train_ts = np.transpose(X_train_raw, (0, 2, 1)).astype(np.float32)
X_test_ts = np.transpose(X_test_raw, (0, 2, 1)).astype(np.float32)

label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train_ts = np.array([label_map[l] for l in y_train_raw], dtype=np.int64)
y_test_ts = np.array([label_map[l] for l in y_test_raw], dtype=np.int64)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_ts), torch.tensor(y_train_ts)), batch_size=8, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test_ts), torch.tensor(y_test_ts)), batch_size=8, shuffle=False)

# Обучение WDWT

print("\nШаг 2: Запуск логов обучения WDWT ...")
start_time = time.time()
model = WDTWClassifier(
    k=3,
    g=0.1
)
y_true = []

model.fit(train_loader)

for _, y_batch in test_loader:
    y_true.extend(
        y_batch.numpy()
    )

y_true = np.array(y_true)

y_pred = model.predict(
    test_loader
)

WDWT_duration = time.time() - start_time
WDWT_accuracy = 100 * accuracy_score(
    y_true,
    y_pred
)

WDWT_precision = precision_score(
        y_true,
        y_pred,
        average="macro"
)

# Обучение ROCKET Baseline
print("\nШаг 3: Обучение baseline-алгоритма ROCKET...")
start_time = time.time()
rocket = Rocket(num_kernels=10000, random_state=42)
X_train_transformed = rocket.fit_transform(X_train_raw)
X_test_transformed = rocket.transform(X_test_raw)
rocket_classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
rocket_classifier.fit(X_train_transformed, y_train_raw)
rocket_duration = time.time() - start_time

y_pred_rocket = rocket_classifier.predict(
    X_test_transformed
)
rocket_accuracy = (
    accuracy_score(
        y_test_raw,
        y_pred_rocket
    ) * 100
)

rocket_precision = precision_score(
        y_test_raw,
        y_pred_rocket,
        average="macro"
)

# Шаг 4: Генерация и сохранение графиков
print("\nШаг 4: Экспорт графиков результатов на диск...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
models = ['ROCKET', 'WDTW']
ax1.bar(models, [rocket_accuracy, WDWT_accuracy], color=['#34495e', '#e67e22'], width=0.5)
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Сравнение точности')
ax1.grid(axis='y', linestyle='--')

ax2.bar(models, [rocket_duration, WDWT_duration], color=['#34495e', '#2ecc71'], width=0.5)
ax2.set_ylabel('Время (сек)')
ax2.set_title('Сравнение скорости обучения')
ax2.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('WDTW_vs_rocket_accuracy.png', dpi=150)
plt.close()

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
models = ['ROCKET', 'WDTW']
ax1.bar(models, [rocket_precision, WDWT_precision], color=['#34495e', '#e67e22'], width=0.5)
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Сравнение точности')
ax1.grid(axis='y', linestyle='--')

ax2.bar(models, [rocket_duration, WDWT_duration], color=['#34495e', '#2ecc71'], width=0.5)
ax2.set_ylabel('Время (сек)')
ax2.set_title('Сравнение скорости обучения')
ax2.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('WDTW_vs_rocket_precision.png', dpi=150)
plt.close()


# Вывод таблицы
print("\n" + "="*55)
print(f"{'МЕТОД':<25} | {'ACCURACY (%)':<12} | {'ВРЕМЯ (сек)':<10}")
print("="*55)
print(f"{'ROCKET (Baseline)':<25} | {rocket_accuracy:<12.2f} | {rocket_duration:<10.3f}")
print(f"{'WDWT (Проект)':<25} | {WDWT_accuracy:<12.2f} | {WDWT_duration:<10.3f}")
print("="*55)
print(f"{'':<25} | {'PRECISION':<12} | {'ВРЕМЯ (сек)':<10}")
print("="*55)
print(f"{'ROCKET (Baseline)':<25} | {rocket_precision:<12.2f} | {rocket_duration:<10.3f}")
print(f"{'WDWT (Проект)':<25} | {WDWT_precision:<12.2f} | {WDWT_duration:<10.3f}")
print("="*55)