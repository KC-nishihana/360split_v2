"""
加速度データから3D軌跡をプロット

IMUの加速度データ（AccelerometerX/Y/Z）から：
1. 重力成分の除去
2. 速度の計算（積分）
3. 位置の計算（積分）
4. 3Dプロット

注意：IMUの積分は誤差が蓄積しやすいため、精度は限定的です。
より高精度な位置推定には、ジャイロデータや外部センサーとの融合が必要です。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from pathlib import Path

# ====== 設定 ======
imu_csv = Path("imu_timeseries.csv")
output_plot = Path("trajectory_3d.png")
output_csv = Path("trajectory_3d.csv")

# フィルタ設定
GRAVITY = 9.81  # 重力加速度 [m/s^2]
LOWPASS_CUTOFF = 0.5  # ローパスフィルタのカットオフ周波数 [Hz]（重力成分推定用）
HIGHPASS_CUTOFF = 0.1  # ハイパスフィルタのカットオフ周波数 [Hz]（ドリフト除去用）

# ====== データ読み込み ======
print(f"[INFO] Loading IMU data from {imu_csv}...")
df = pd.read_csv(imu_csv)

# 必要な列のチェック
required_cols = ["TimeStamp0", "AccelerometerX", "AccelerometerY", "AccelerometerZ"]
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"Required columns not found: {required_cols}")

# データの取得
time = df["TimeStamp0"].values
acc_x = df["AccelerometerX"].values
acc_y = df["AccelerometerY"].values
acc_z = df["AccelerometerZ"].values

print(f"[INFO] Loaded {len(time)} samples")
print(f"[INFO] Time range: {time[0]:.3f} - {time[-1]:.3f} seconds")
print(f"[INFO] Sample rate: {len(time) / (time[-1] - time[0]):.1f} Hz")

# ====== 重力成分の除去 ======
print(f"\n[INFO] Removing gravity component...")

# サンプリング周波数の計算
fs = len(time) / (time[-1] - time[0])  # Hz

# ローパスフィルタで重力成分を推定
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

# 重力成分（ローパスフィルタで推定）
gravity_x = butter_lowpass_filter(acc_x, LOWPASS_CUTOFF, fs)
gravity_y = butter_lowpass_filter(acc_y, LOWPASS_CUTOFF, fs)
gravity_z = butter_lowpass_filter(acc_z, LOWPASS_CUTOFF, fs)

# 動的加速度成分（重力を除去）
acc_dynamic_x = acc_x - gravity_x
acc_dynamic_y = acc_y - gravity_y
acc_dynamic_z = acc_z - gravity_z

print(f"[INFO] Gravity estimate: X={gravity_x.mean():.3f}, Y={gravity_y.mean():.3f}, Z={gravity_z.mean():.3f} [g]")
print(f"[INFO] Dynamic acceleration RMS: X={np.sqrt(np.mean(acc_dynamic_x**2)):.4f}, "
      f"Y={np.sqrt(np.mean(acc_dynamic_y**2)):.4f}, Z={np.sqrt(np.mean(acc_dynamic_z**2)):.4f} [g]")

# ====== 単位変換 ======
# 加速度を g から m/s^2 に変換
acc_dynamic_x_ms2 = acc_dynamic_x * GRAVITY
acc_dynamic_y_ms2 = acc_dynamic_y * GRAVITY
acc_dynamic_z_ms2 = acc_dynamic_z * GRAVITY

# ====== 速度の計算（積分） ======
print(f"\n[INFO] Calculating velocity (integration)...")

# 台形則による積分
dt = np.diff(time)
vel_x = np.zeros_like(time)
vel_y = np.zeros_like(time)
vel_z = np.zeros_like(time)

for i in range(1, len(time)):
    vel_x[i] = vel_x[i-1] + 0.5 * (acc_dynamic_x_ms2[i-1] + acc_dynamic_x_ms2[i]) * dt[i-1]
    vel_y[i] = vel_y[i-1] + 0.5 * (acc_dynamic_y_ms2[i-1] + acc_dynamic_y_ms2[i]) * dt[i-1]
    vel_z[i] = vel_z[i-1] + 0.5 * (acc_dynamic_z_ms2[i-1] + acc_dynamic_z_ms2[i]) * dt[i-1]

print(f"[INFO] Velocity range: X=[{vel_x.min():.3f}, {vel_x.max():.3f}], "
      f"Y=[{vel_y.min():.3f}, {vel_y.max():.3f}], "
      f"Z=[{vel_z.min():.3f}, {vel_z.max():.3f}] [m/s]")

# ドリフト除去（オプション：ハイパスフィルタ）
def butter_highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

# 速度のドリフトを除去
vel_x = butter_highpass_filter(vel_x, HIGHPASS_CUTOFF, fs)
vel_y = butter_highpass_filter(vel_y, HIGHPASS_CUTOFF, fs)
vel_z = butter_highpass_filter(vel_z, HIGHPASS_CUTOFF, fs)

print(f"[INFO] Velocity after drift removal: X=[{vel_x.min():.3f}, {vel_x.max():.3f}], "
      f"Y=[{vel_y.min():.3f}, {vel_y.max():.3f}], "
      f"Z=[{vel_z.min():.3f}, {vel_z.max():.3f}] [m/s]")

# ====== 位置の計算（積分） ======
print(f"\n[INFO] Calculating position (integration)...")

pos_x = np.zeros_like(time)
pos_y = np.zeros_like(time)
pos_z = np.zeros_like(time)

for i in range(1, len(time)):
    pos_x[i] = pos_x[i-1] + 0.5 * (vel_x[i-1] + vel_x[i]) * dt[i-1]
    pos_y[i] = pos_y[i-1] + 0.5 * (vel_y[i-1] + vel_y[i]) * dt[i-1]
    pos_z[i] = pos_z[i-1] + 0.5 * (vel_z[i-1] + vel_z[i]) * dt[i-1]

print(f"[INFO] Position range: X=[{pos_x.min():.3f}, {pos_x.max():.3f}], "
      f"Y=[{pos_y.min():.3f}, {pos_y.max():.3f}], "
      f"Z=[{pos_z.min():.3f}, {pos_z.max():.3f}] [m]")

# 総移動距離の計算
displacement = np.sqrt(np.diff(pos_x)**2 + np.diff(pos_y)**2 + np.diff(pos_z)**2)
total_distance = np.sum(displacement)
print(f"[INFO] Total path length: {total_distance:.3f} meters")
print(f"[INFO] Straight-line distance: {np.sqrt(pos_x[-1]**2 + pos_y[-1]**2 + pos_z[-1]**2):.3f} meters")

# ====== CSV出力 ======
output_df = pd.DataFrame({
    "TimeStamp0": time,
    "AccelX_g": acc_x,
    "AccelY_g": acc_y,
    "AccelZ_g": acc_z,
    "AccelDynamicX_ms2": acc_dynamic_x_ms2,
    "AccelDynamicY_ms2": acc_dynamic_y_ms2,
    "AccelDynamicZ_ms2": acc_dynamic_z_ms2,
    "VelocityX_ms": vel_x,
    "VelocityY_ms": vel_y,
    "VelocityZ_ms": vel_z,
    "PositionX_m": pos_x,
    "PositionY_m": pos_y,
    "PositionZ_m": pos_z,
})

output_df.to_csv(output_csv, index=False)
print(f"\n[OK] Saved trajectory data -> {output_csv}")

# ====== 3Dプロット ======
print(f"\n[INFO] Creating 3D trajectory plot...")

fig = plt.figure(figsize=(14, 10))

# サブプロット1: 3D軌跡
ax1 = fig.add_subplot(221, projection='3d')
# 時間に応じた色付け
colors = plt.cm.viridis(np.linspace(0, 1, len(time)))
for i in range(len(time) - 1):
    ax1.plot(pos_x[i:i+2], pos_y[i:i+2], pos_z[i:i+2],
             color=colors[i], linewidth=1.5)

# 開始点と終了点をマーク
ax1.scatter(pos_x[0], pos_y[0], pos_z[0], c='green', s=100, marker='o', label='Start')
ax1.scatter(pos_x[-1], pos_y[-1], pos_z[-1], c='red', s=100, marker='x', label='End')

ax1.set_xlabel('X Position [m]')
ax1.set_ylabel('Y Position [m]')
ax1.set_zlabel('Z Position [m]')
ax1.set_title('3D Trajectory (Color: Time)')
ax1.legend()
ax1.grid(True)

# サブプロット2: XY平面
ax2 = fig.add_subplot(222)
scatter = ax2.scatter(pos_x, pos_y, c=time, cmap='viridis', s=10)
ax2.plot(pos_x[0], pos_y[0], 'go', markersize=10, label='Start')
ax2.plot(pos_x[-1], pos_y[-1], 'rx', markersize=10, label='End')
ax2.set_xlabel('X Position [m]')
ax2.set_ylabel('Y Position [m]')
ax2.set_title('XY Plane Trajectory')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')
plt.colorbar(scatter, ax=ax2, label='Time [s]')

# サブプロット3: XZ平面
ax3 = fig.add_subplot(223)
scatter = ax3.scatter(pos_x, pos_z, c=time, cmap='viridis', s=10)
ax3.plot(pos_x[0], pos_z[0], 'go', markersize=10, label='Start')
ax3.plot(pos_x[-1], pos_z[-1], 'rx', markersize=10, label='End')
ax3.set_xlabel('X Position [m]')
ax3.set_ylabel('Z Position [m]')
ax3.set_title('XZ Plane Trajectory')
ax3.legend()
ax3.grid(True)
ax3.axis('equal')
plt.colorbar(scatter, ax=ax3, label='Time [s]')

# サブプロット4: YZ平面
ax4 = fig.add_subplot(224)
scatter = ax4.scatter(pos_y, pos_z, c=time, cmap='viridis', s=10)
ax4.plot(pos_y[0], pos_z[0], 'go', markersize=10, label='Start')
ax4.plot(pos_y[-1], pos_z[-1], 'rx', markersize=10, label='End')
ax4.set_xlabel('Y Position [m]')
ax4.set_ylabel('Z Position [m]')
ax4.set_title('YZ Plane Trajectory')
ax4.legend()
ax4.grid(True)
ax4.axis('equal')
plt.colorbar(scatter, ax=ax4, label='Time [s]')

plt.tight_layout()
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"[OK] Saved 3D plot -> {output_plot}")

# ====== 追加プロット: 速度と加速度の時系列 ======
fig2, axes = plt.subplots(3, 1, figsize=(12, 10))

# 加速度
axes[0].plot(time, acc_dynamic_x_ms2, label='X', alpha=0.7)
axes[0].plot(time, acc_dynamic_y_ms2, label='Y', alpha=0.7)
axes[0].plot(time, acc_dynamic_z_ms2, label='Z', alpha=0.7)
axes[0].set_ylabel('Acceleration [m/s²]')
axes[0].set_title('Dynamic Acceleration (Gravity Removed)')
axes[0].legend()
axes[0].grid(True)

# 速度
axes[1].plot(time, vel_x, label='X', alpha=0.7)
axes[1].plot(time, vel_y, label='Y', alpha=0.7)
axes[1].plot(time, vel_z, label='Z', alpha=0.7)
axes[1].set_ylabel('Velocity [m/s]')
axes[1].set_title('Velocity')
axes[1].legend()
axes[1].grid(True)

# 位置
axes[2].plot(time, pos_x, label='X', alpha=0.7)
axes[2].plot(time, pos_y, label='Y', alpha=0.7)
axes[2].plot(time, pos_z, label='Z', alpha=0.7)
axes[2].set_xlabel('Time [s]')
axes[2].set_ylabel('Position [m]')
axes[2].set_title('Position')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("trajectory_timeseries.png", dpi=150, bbox_inches='tight')
print(f"[OK] Saved time series plot -> trajectory_timeseries.png")

print(f"\n=== Summary ===")
print(f"Total samples: {len(time)}")
print(f"Duration: {time[-1] - time[0]:.2f} seconds")
print(f"Total path length: {total_distance:.3f} meters")
print(f"Straight-line distance: {np.sqrt(pos_x[-1]**2 + pos_y[-1]**2 + pos_z[-1]**2):.3f} meters")
print(f"\nOutput files:")
print(f"  - {output_plot}")
print(f"  - trajectory_timeseries.png")
print(f"  - {output_csv}")
print(f"\n[NOTE] IMU-based position estimation has limited accuracy due to drift.")
print(f"[NOTE] For better results, consider sensor fusion with gyroscope and external references.")

plt.show()
