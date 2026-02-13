"""
Visual Odometry Only - Test Version

左映像からのモノキュラーVOをテスト
既存のGeometricEvaluatorを活用
"""

import sys
sys.path.insert(0, '/sessions/wizardly-gallant-johnson/mnt/360split')

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 既存モジュールのインポート
from core.geometric_evaluator import GeometricEvaluator


class SimpleVisualOdometry:
    """シンプルなモノキュラーVisual Odometry"""

    def __init__(self):
        """初期化"""
        print("[INFO] Initializing Visual Odometry...")
        self.geo_eval = GeometricEvaluator(use_sift=False)

        # カメラ行列（仮定）
        focal_length = 3840 * 0.5
        cx, cy = 3840 / 2, 3840 / 2

        self.K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # 状態
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))
        self.trajectory = [self.t_total.copy()]

        self.prev_frame = None
        self.prev_idx = None

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> dict:
        """フレーム処理"""
        if self.prev_frame is None:
            self.prev_frame = frame
            self.prev_idx = frame_idx
            return {'status': 'init', 'position': self.t_total.flatten()}

        # 特徴点検出
        kp1, desc1 = self.geo_eval._detect_and_compute_cached(
            self.prev_frame, frame_idx=self.prev_idx, use_polar_mask=True
        )
        kp2, desc2 = self.geo_eval._detect_and_compute_cached(
            frame, frame_idx=frame_idx, use_polar_mask=True
        )

        # マッチング
        matches = self.geo_eval._match_features(desc1, desc2, kp1, kp2)

        if len(matches) < 8:
            self.prev_frame = frame
            self.prev_idx = frame_idx
            return {'status': 'insufficient_matches', 'matches': len(matches)}

        # 対応点
        pts1 = np.float32([kp1[m[0]].pt for m in matches])
        pts2 = np.float32([kp2[m[1]].pt for m in matches])

        # Essential Matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K,
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None or mask is None:
            self.prev_frame = frame
            self.prev_idx = frame_idx
            return {'status': 'estimation_failed', 'matches': len(matches)}

        # R, t 復元
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # 累積
        self.t_total = self.t_total + self.R_total @ t
        self.R_total = R @ self.R_total
        self.trajectory.append(self.t_total.copy())

        # 更新
        self.prev_frame = frame
        self.prev_idx = frame_idx

        return {
            'status': 'success',
            'matches': len(matches),
            'inliers': int(np.sum(mask)),
            'position': self.t_total.flatten()
        }

    def get_trajectory(self) -> np.ndarray:
        """軌跡取得"""
        return np.array([t.flatten() for t in self.trajectory])


def plot_trajectory(trajectory: np.ndarray, output_path: Path):
    """軌跡プロット"""
    if len(trajectory) < 2:
        print("[WARN] Not enough points to plot")
        return

    fig = plt.figure(figsize=(16, 10))

    # 3D
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            'b-', linewidth=2, label='VO Trajectory')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
               c='g', s=100, marker='o', label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
               c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory (Visual Odometry)')
    ax1.legend()
    ax1.grid(True)

    # XY
    ax2 = fig.add_subplot(222)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='g', s=100, label='Start')
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='r', s=100, marker='x', label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # XZ
    ax3 = fig.add_subplot(223)
    ax3.plot(trajectory[:, 0], trajectory[:, 2], 'b-', linewidth=2)
    ax3.scatter(trajectory[0, 0], trajectory[0, 2], c='g', s=100, label='Start')
    ax3.scatter(trajectory[-1, 0], trajectory[-1, 2], c='r', s=100, marker='x', label='End')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Plane')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')

    # YZ
    ax4 = fig.add_subplot(224)
    ax4.plot(trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2)
    ax4.scatter(trajectory[0, 1], trajectory[0, 2], c='g', s=100, label='Start')
    ax4.scatter(trajectory[-1, 1], trajectory[-1, 2], c='r', s=100, marker='x', label='End')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ Plane')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved -> {output_path}")


def main():
    """メイン処理"""
    print("="*80)
    print("Visual Odometry Test (Monocular)")
    print("="*80)
    print()

    video_path = Path("left_eye.mp4")
    output_plot = Path("vo_trajectory_test.png")

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return

    vo = SimpleVisualOdometry()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total frames: {frame_count}")
    print(f"[INFO] Processing every 30th frame (first 300 frames only)...")
    print()

    frame_skip = 30
    max_frames = 300  # 最初の300フレームのみ処理
    frame_idx = 0
    processed = 0
    success_count = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            result = vo.process_frame(frame, frame_idx)
            processed += 1

            if result['status'] == 'success':
                success_count += 1

            if processed % 20 == 0:
                print(f"  Frame {frame_idx}/{frame_count}: "
                      f"status={result['status']}, "
                      f"matches={result.get('matches', 0)}, "
                      f"pos={result.get('position', [0,0,0])[:2]}")

        frame_idx += 1

    cap.release()

    print()
    print(f"[INFO] Processed: {processed} frames")
    print(f"[INFO] Successful: {success_count} frames")

    # 軌跡取得
    trajectory = vo.get_trajectory()
    print(f"[INFO] Trajectory points: {len(trajectory)}")

    if len(trajectory) > 1:
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        total_dist = np.sum(distances)
        final_pos = trajectory[-1]
        straight_dist = np.linalg.norm(final_pos)

        print()
        print("="*80)
        print("Results")
        print("="*80)
        print(f"Total path length: {total_dist:.3f} (arbitrary units)")
        print(f"Straight-line distance: {straight_dist:.3f}")
        print(f"Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
        print()
        print("[NOTE] Scale is relative (monocular VO limitation)")
        print("[NOTE] Absolute scale requires IMU fusion or known distances")

    # プロット
    print("\n[INFO] Generating plot...")
    plot_trajectory(trajectory, output_plot)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
