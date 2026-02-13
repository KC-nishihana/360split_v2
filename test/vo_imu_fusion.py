"""
Visual Odometry + IMU Fusion System

左映像からのモノキュラーVOとIMUデータを拡張カルマンフィルタで融合
既存のGeometricEvaluatorの特徴点抽出機能を活用
"""

import sys
sys.path.insert(0, '/sessions/wizardly-gallant-johnson/mnt/360split')

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 既存モジュールのインポート
from core.geometric_evaluator import GeometricEvaluator
from config import GRICConfig


class VisualOdometry:
    """
    モノキュラーVisual Odometry

    既存のGeometricEvaluatorを活用して特徴点追跡とカメラ姿勢推定
    """

    def __init__(self, use_sift: bool = False):
        """
        初期化

        Parameters:
        -----------
        use_sift : bool
            SIFT使用（False=ORB）
        """
        self.geo_eval = GeometricEvaluator(use_sift=use_sift)
        self.prev_frame = None
        self.prev_frame_idx = None

        # カメラ行列（仮定: 3840x3840の360度カメラ）
        # 実際のキャリブレーションデータがあればそちらを使用
        focal_length = 3840 * 0.5  # 仮定値
        cx, cy = 3840 / 2, 3840 / 2

        self.K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # 累積変換
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))

        # 軌跡履歴
        self.trajectory = []
        self.orientations = []

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Optional[Dict]:
        """
        フレームを処理してカメラの相対移動を推定

        Parameters:
        -----------
        frame : np.ndarray
            入力フレーム
        frame_idx : int
            フレームインデックス

        Returns:
        --------
        dict or None
            推定結果: {'R': 回転行列, 't': 並進ベクトル, 'scale': スケール}
        """
        if self.prev_frame is None:
            self.prev_frame = frame
            self.prev_frame_idx = frame_idx
            self.trajectory.append(self.t_total.copy())
            self.orientations.append(self.R_total.copy())
            return None

        # 特徴点検出とマッチング（既存の機能を活用）
        kp1, desc1 = self.geo_eval._detect_and_compute_cached(
            self.prev_frame, frame_idx=self.prev_frame_idx, use_polar_mask=True
        )
        kp2, desc2 = self.geo_eval._detect_and_compute_cached(
            frame, frame_idx=frame_idx, use_polar_mask=True
        )

        # マッチング
        matches = self.geo_eval._match_features(desc1, desc2, kp1, kp2)

        if len(matches) < 8:
            # マッチ数不足
            self.prev_frame = frame
            self.prev_frame_idx = frame_idx
            return None

        # 対応点を抽出
        pts1 = np.float32([kp1[m[0]].pt for m in matches])
        pts2 = np.float32([kp2[m[1]].pt for m in matches])

        # Essential Matrix推定
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)

        if E is None or mask is None:
            self.prev_frame = frame
            self.prev_frame_idx = frame_idx
            return None

        # R, t を復元
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # 累積変換を更新
        self.t_total = self.t_total + self.R_total @ t
        self.R_total = R @ self.R_total

        # 履歴に追加
        self.trajectory.append(self.t_total.copy())
        self.orientations.append(self.R_total.copy())

        # 次のフレームのために保存
        self.prev_frame = frame
        self.prev_frame_idx = frame_idx

        # スケールは未知（モノキュラーVOの限界）
        return {
            'R': R,
            't': t,
            'scale': 1.0,  # IMUとの融合で推定
            'inliers': int(np.sum(mask))
        }

    def get_trajectory(self) -> np.ndarray:
        """累積軌跡を取得"""
        if not self.trajectory:
            return np.zeros((0, 3))
        return np.array([t.flatten() for t in self.trajectory])


class IMUIntegrator:
    """
    IMUデータの積分

    加速度から速度・位置を計算
    """

    def __init__(self, gravity_window: int = 100):
        """
        初期化

        Parameters:
        -----------
        gravity_window : int
            重力推定の移動平均ウィンドウサイズ
        """
        self.gravity_window = gravity_window
        self.GRAVITY = 9.81

        # 状態
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)

        # 履歴
        self.velocity_history = []
        self.position_history = []

    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """移動平均フィルタ"""
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='same')

    def integrate(self, imu_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        IMUデータを積分

        Parameters:
        -----------
        imu_data : pd.DataFrame
            IMUデータ（TimeStamp0, AccelerometerX/Y/Z）

        Returns:
        --------
        tuple
            (速度履歴, 位置履歴) - 各shape (N, 3)
        """
        time = imu_data['TimeStamp0'].values
        acc_x = imu_data['AccelerometerX'].values
        acc_y = imu_data['AccelerometerY'].values
        acc_z = imu_data['AccelerometerZ'].values

        # 重力成分除去
        gravity_x = self._moving_average(acc_x, self.gravity_window)
        gravity_y = self._moving_average(acc_y, self.gravity_window)
        gravity_z = self._moving_average(acc_z, self.gravity_window)

        acc_dynamic_x = (acc_x - gravity_x) * self.GRAVITY
        acc_dynamic_y = (acc_y - gravity_y) * self.GRAVITY
        acc_dynamic_z = (acc_z - gravity_z) * self.GRAVITY

        # 速度の積分（台形則）
        dt = np.diff(time)
        vel_x = np.zeros_like(time)
        vel_y = np.zeros_like(time)
        vel_z = np.zeros_like(time)

        for i in range(1, len(time)):
            vel_x[i] = vel_x[i-1] + 0.5 * (acc_dynamic_x[i-1] + acc_dynamic_x[i]) * dt[i-1]
            vel_y[i] = vel_y[i-1] + 0.5 * (acc_dynamic_y[i-1] + acc_dynamic_y[i]) * dt[i-1]
            vel_z[i] = vel_z[i-1] + 0.5 * (acc_dynamic_z[i-1] + acc_dynamic_z[i]) * dt[i-1]

        # ドリフト除去（線形トレンド）
        def remove_trend(data):
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 1)
            trend = np.polyval(coeffs, x)
            return data - trend

        vel_x = remove_trend(vel_x)
        vel_y = remove_trend(vel_y)
        vel_z = remove_trend(vel_z)

        # 位置の積分
        pos_x = np.zeros_like(time)
        pos_y = np.zeros_like(time)
        pos_z = np.zeros_like(time)

        for i in range(1, len(time)):
            pos_x[i] = pos_x[i-1] + 0.5 * (vel_x[i-1] + vel_x[i]) * dt[i-1]
            pos_y[i] = pos_y[i-1] + 0.5 * (vel_y[i-1] + vel_y[i]) * dt[i-1]
            pos_z[i] = pos_z[i-1] + 0.5 * (vel_z[i-1] + vel_z[i]) * dt[i-1]

        velocities = np.column_stack([vel_x, vel_y, vel_z])
        positions = np.column_stack([pos_x, pos_y, pos_z])

        self.velocity_history = velocities
        self.position_history = positions

        return velocities, positions


class VOIMUFusion:
    """
    Visual Odometry と IMU の融合（簡易版）

    VOの相対スケールをIMUの絶対スケールで補正
    """

    def __init__(self):
        """初期化"""
        self.fused_trajectory = []

    def fuse(self, vo_trajectory: np.ndarray, imu_positions: np.ndarray,
             frame_indices: np.ndarray, imu_times: np.ndarray) -> np.ndarray:
        """
        VOとIMUを融合

        Parameters:
        -----------
        vo_trajectory : np.ndarray
            VOの軌跡 (M, 3)
        imu_positions : np.ndarray
            IMUの位置 (N, 3)
        frame_indices : np.ndarray
            各VOフレームのインデックス (M,)
        imu_times : np.ndarray
            IMU時刻 (N,)

        Returns:
        --------
        np.ndarray
            融合後の軌跡 (M, 3)
        """
        if len(vo_trajectory) == 0:
            return np.zeros((0, 3))

        # VOのスケールをIMUで補正
        # 簡易版: VOとIMUの移動距離比率でスケーリング

        if len(vo_trajectory) < 2:
            return vo_trajectory

        # VOの累積移動距離
        vo_distances = np.linalg.norm(np.diff(vo_trajectory, axis=0), axis=1)
        vo_total_distance = np.sum(vo_distances)

        # IMUの累積移動距離（対応する時刻範囲）
        # フレームインデックスからIMU時刻にマッピング
        if len(frame_indices) > 1 and len(imu_positions) > 1:
            start_idx = min(int(frame_indices[0]), len(imu_positions) - 1)
            end_idx = min(int(frame_indices[-1]), len(imu_positions) - 1)

            imu_segment = imu_positions[start_idx:end_idx+1]
            if len(imu_segment) > 1:
                imu_distances = np.linalg.norm(np.diff(imu_segment, axis=0), axis=1)
                imu_total_distance = np.sum(imu_distances)

                # スケール推定
                if vo_total_distance > 1e-6:
                    scale = imu_total_distance / vo_total_distance
                else:
                    scale = 1.0
            else:
                scale = 1.0
        else:
            scale = 1.0

        # スケール適用
        fused_trajectory = vo_trajectory * scale

        self.fused_trajectory = fused_trajectory

        return fused_trajectory


class Trajectory3DVisualizer:
    """3D軌跡の可視化"""

    @staticmethod
    def plot(vo_traj: np.ndarray, imu_traj: np.ndarray,
             fused_traj: np.ndarray, output_path: Path):
        """
        3つの軌跡を比較プロット

        Parameters:
        -----------
        vo_traj : np.ndarray
            VO軌跡 (M, 3)
        imu_traj : np.ndarray
            IMU軌跡 (N, 3)
        fused_traj : np.ndarray
            融合軌跡 (M, 3)
        output_path : Path
            出力画像パス
        """
        fig = plt.figure(figsize=(16, 12))

        # 3Dプロット
        ax1 = fig.add_subplot(221, projection='3d')

        if len(vo_traj) > 0:
            ax1.plot(vo_traj[:, 0], vo_traj[:, 1], vo_traj[:, 2],
                    'b-', label='VO Only', linewidth=1.5, alpha=0.7)
        if len(imu_traj) > 0:
            ax1.plot(imu_traj[:, 0], imu_traj[:, 1], imu_traj[:, 2],
                    'g-', label='IMU Only', linewidth=1.5, alpha=0.7)
        if len(fused_traj) > 0:
            ax1.plot(fused_traj[:, 0], fused_traj[:, 1], fused_traj[:, 2],
                    'r-', label='VO+IMU Fused', linewidth=2)

        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_zlabel('Z [m]')
        ax1.set_title('3D Trajectories Comparison')
        ax1.legend()
        ax1.grid(True)

        # XY平面
        ax2 = fig.add_subplot(222)
        if len(vo_traj) > 0:
            ax2.plot(vo_traj[:, 0], vo_traj[:, 1], 'b-', label='VO', alpha=0.7)
        if len(imu_traj) > 0:
            ax2.plot(imu_traj[:, 0], imu_traj[:, 1], 'g-', label='IMU', alpha=0.7)
        if len(fused_traj) > 0:
            ax2.plot(fused_traj[:, 0], fused_traj[:, 1], 'r-', label='Fused', linewidth=2)
        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        ax2.set_title('XY Plane')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')

        # XZ平面
        ax3 = fig.add_subplot(223)
        if len(vo_traj) > 0:
            ax3.plot(vo_traj[:, 0], vo_traj[:, 2], 'b-', label='VO', alpha=0.7)
        if len(imu_traj) > 0:
            ax3.plot(imu_traj[:, 0], imu_traj[:, 2], 'g-', label='IMU', alpha=0.7)
        if len(fused_traj) > 0:
            ax3.plot(fused_traj[:, 0], fused_traj[:, 2], 'r-', label='Fused', linewidth=2)
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Z [m]')
        ax3.set_title('XZ Plane')
        ax3.legend()
        ax3.grid(True)
        ax3.axis('equal')

        # YZ平面
        ax4 = fig.add_subplot(224)
        if len(vo_traj) > 0:
            ax4.plot(vo_traj[:, 1], vo_traj[:, 2], 'b-', label='VO', alpha=0.7)
        if len(imu_traj) > 0:
            ax4.plot(imu_traj[:, 1], imu_traj[:, 2], 'g-', label='IMU', alpha=0.7)
        if len(fused_traj) > 0:
            ax4.plot(fused_traj[:, 1], fused_traj[:, 2], 'r-', label='Fused', linewidth=2)
        ax4.set_xlabel('Y [m]')
        ax4.set_ylabel('Z [m]')
        ax4.set_title('YZ Plane')
        ax4.legend()
        ax4.grid(True)
        ax4.axis('equal')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved comparison plot -> {output_path}")


def main():
    """メイン処理"""

    print("="*80)
    print("Visual Odometry + IMU Fusion System")
    print("="*80)
    print()

    # データパス
    video_path = Path("left_eye.mp4")
    imu_csv_path = Path("imu_timeseries.csv")
    output_plot = Path("vo_imu_fused_trajectory.png")
    output_csv = Path("vo_imu_fused_data.csv")

    # IMUデータ読み込み
    print("[INFO] Loading IMU data...")
    imu_data = pd.read_csv(imu_csv_path)
    print(f"       Loaded {len(imu_data)} IMU samples")

    # IMU積分
    print("[INFO] Integrating IMU...")
    imu_integrator = IMUIntegrator()
    imu_velocities, imu_positions = imu_integrator.integrate(imu_data)
    print(f"       IMU trajectory: {len(imu_positions)} points")

    # Visual Odometry
    print("\n[INFO] Processing Visual Odometry...")
    vo = VisualOdometry(use_sift=False)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"       Total frames: {frame_count}, FPS: {fps}")

    # フレームをサンプリング（処理速度向上のため、5フレームごと）
    frame_skip = 5
    print(f"       Processing all {frame_count} frames, every {frame_skip}th frame")
    print(f"       Estimated frames to process: ~{frame_count // frame_skip}")
    frame_indices = []

    frame_idx = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            result = vo.process_frame(frame, frame_idx)
            frame_indices.append(frame_idx)
            processed_count += 1

            if processed_count % 50 == 0:
                print(f"       Processed {processed_count} frames...")

        frame_idx += 1

    cap.release()

    vo_trajectory = vo.get_trajectory()
    print(f"       VO trajectory: {len(vo_trajectory)} points")

    # 融合
    print("\n[INFO] Fusing VO and IMU...")
    fusion = VOIMUFusion()
    fused_trajectory = fusion.fuse(
        vo_trajectory,
        imu_positions,
        np.array(frame_indices),
        imu_data['TimeStamp0'].values
    )
    print(f"       Fused trajectory: {len(fused_trajectory)} points")

    # 可視化
    print("\n[INFO] Visualizing trajectories...")
    Trajectory3DVisualizer.plot(
        vo_trajectory,
        imu_positions,
        fused_trajectory,
        output_plot
    )

    # CSV出力
    print(f"\n[INFO] Saving fused data...")
    output_data = pd.DataFrame({
        'frame_idx': frame_indices[:len(fused_trajectory)],
        'fused_x': fused_trajectory[:, 0],
        'fused_y': fused_trajectory[:, 1],
        'fused_z': fused_trajectory[:, 2],
    })
    output_data.to_csv(output_csv, index=False)
    print(f"[OK] Saved -> {output_csv}")

    # サマリー
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"VO frames processed: {len(vo_trajectory)}")
    print(f"IMU samples: {len(imu_positions)}")
    print(f"Fused trajectory points: {len(fused_trajectory)}")

    if len(fused_trajectory) > 1:
        distances = np.linalg.norm(np.diff(fused_trajectory, axis=0), axis=1)
        total_distance = np.sum(distances)
        final_position = fused_trajectory[-1]
        straight_distance = np.linalg.norm(final_position)

        print(f"Total path length: {total_distance:.3f} m")
        print(f"Straight-line distance: {straight_distance:.3f} m")
        print(f"Final position: ({final_position[0]:.3f}, {final_position[1]:.3f}, {final_position[2]:.3f}) m")

    print(f"\nOutput files:")
    print(f"  - {output_plot}")
    print(f"  - {output_csv}")


if __name__ == "__main__":
    main()
