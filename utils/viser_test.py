import time
import numpy as np
import viser


POSE_CONNECTIONS = [
    (11, 12), (12, 14), (14, 16), (11, 13), (13, 15),  # arms
    (11, 23), (12, 24), (23, 24),                     # torso
    (23, 25), (25, 27), (24, 26), (26, 28),           # legs
    (27, 29), (29, 31), (28, 30), (30, 32)            # lower legs
]


class PoseViser:
    def __init__(self, grid_size=2.0, grid_spacing=0.2, fps=30):
        self.fps = fps
        self.server = viser.ViserServer()
        print("Open browser → http://localhost:4242")

        self._add_ground_grid(grid_size, grid_spacing)
        self._init_pose_handles()

    # ---------------------------------
    # 1) Ground Grid
    # ---------------------------------
    def _add_ground_grid(self, grid_size, grid_spacing):
        x = np.arange(-grid_size, grid_size + grid_spacing, grid_spacing)
        z = np.arange(-grid_size, grid_size + grid_spacing, grid_spacing)

        vertical = np.array([[[xi, 0, -grid_size], [xi, 0, grid_size]] for xi in x])
        horizontal = np.array([[[-grid_size, 0, zi], [grid_size, 0, zi]] for zi in z])
        grid = np.concatenate([vertical, horizontal], axis=0)

        colors = np.tile(np.array([[0.8, 0.8, 0.8]]), (len(grid), 2, 1))

        self.server.scene.add_line_segments(
            name="/ground_grid",
            points=grid,
            colors=colors,
            line_width=2.0,
        )

    # ---------------------------------
    # 2) Init Pose Handles
    # ---------------------------------
    def _init_pose_handles(self):
        n_points = 33

        # point cloud for joints
        init_pts = np.zeros((n_points, 3))
        init_colors = np.tile(np.array([[0.2, 0.7, 1.0]]), (n_points, 1))

        self.pose_points = self.server.scene.add_point_cloud(
            name="/pose_points",
            points=init_pts,
            colors=init_colors,
            point_size=0.03,
        )

        # line segments for skeleton
        seg_points = np.zeros((len(POSE_CONNECTIONS), 2, 3))
        seg_colors = np.tile(np.array([[1.0, 0.8, 0.2]]), (len(seg_points), 2, 1))

        self.pose_lines = self.server.scene.add_line_segments(
            name="/pose_lines",
            points=seg_points,
            colors=seg_colors,
            line_width=3.0,
        )

    # ---------------------------------
    # 3) Update a Single Frame
    # ---------------------------------
    def _update_frame(self, pts):
        """
        pts : (33,3)
        """
        self.pose_points.points = pts

        seg = np.array([[pts[a], pts[b]] for (a, b) in POSE_CONNECTIONS])
        self.pose_lines.points = seg

    # ---------------------------------
    # 4) Play Full Sequence
    # ---------------------------------
    def play_sequence(self, seq):
        """
        seq: (T,33,3)
        """
        print("▶ Playing sequence...")

        try:
            while True:
                for t in range(seq.shape[0]):
                    self._update_frame(seq[t])
                    time.sleep(1 / self.fps)
        except KeyboardInterrupt:
            print("🛑 Stopped.")


    # ---------------------------------
    # 5) Update a Dual Frame
    # ---------------------------------
    def _init_dual_handles(self, offset=1.5):
        """
        offset: 두 skeleton의 간격 (x축 방향 이동)
        """
        n_points = 33

        # -------- Raw skeleton (Left) --------
        init_pts = np.zeros((n_points, 3))
        raw_colors = np.tile(np.array([[0.2, 0.7, 1.0]]), (n_points, 1))

        self.raw_points = self.server.scene.add_point_cloud(
            name="/raw_points",
            points=init_pts,
            colors=raw_colors,
            point_size=0.03,
        )

        raw_seg_pts = np.zeros((len(POSE_CONNECTIONS), 2, 3))
        raw_seg_colors = np.tile(np.array([[1.0, 0.5, 0.2]]), (len(raw_seg_pts), 2, 1))

        self.raw_lines = self.server.scene.add_line_segments(
            name="/raw_lines",
            points=raw_seg_pts,
            colors=raw_seg_colors,
            line_width=3.0,
        )

        # -------- Refined skeleton (Right) --------
        refined_pts = np.zeros((n_points, 3))
        refined_colors = np.tile(np.array([[0.9, 0.7, 1.0]]), (n_points, 1))

        self.refined_points = self.server.scene.add_point_cloud(
            name="/refined_points",
            points=refined_pts,
            colors=refined_colors,
            point_size=0.03,
        )

        refined_seg_pts = np.zeros((len(POSE_CONNECTIONS), 2, 3))
        refined_seg_colors = np.tile(np.array([[0.5, 1.0, 0.3]]), (len(refined_seg_pts), 2, 1))

        self.refined_lines = self.server.scene.add_line_segments(
            name="/refined_lines",
            points=refined_seg_pts,
            colors=refined_seg_colors,
            line_width=3.0,
        )

        self.offset = offset
    def _update_dual_frame(self, raw_pts, refined_pts):
        """
        raw_pts:     (33,3)
        refined_pts: (33,3)
        """

        # Left: Raw
        self.raw_points.points = raw_pts
        raw_seg = np.array([[raw_pts[a], raw_pts[b]] for (a, b) in POSE_CONNECTIONS])
        self.raw_lines.points = raw_seg

        # Right: Refined (x축으로 offset)
        refined_shifted = refined_pts.copy()
        refined_shifted[:, 0] += self.offset   # move to the right

        self.refined_points.points = refined_shifted
        refined_seg = np.array([[refined_shifted[a], refined_shifted[b]] for (a, b) in POSE_CONNECTIONS])
        self.refined_lines.points = refined_seg

    # ---------------------------------
    # 6) Play Full Sequence
    # ---------------------------------
    def play_two_sequences(self, raw_seq, refined_seq,offset=1.5):
        """
        raw_seq:     (T,33,3)
        refined_seq: (T,33,3)
        """

        assert raw_seq.shape == refined_seq.shape

        self._init_dual_handles(offset=offset)

        print("▶ Playing RAW (left) vs REFINED (right) ...")

        T = raw_seq.shape[0]

        try:
            while True:
                for t in range(T):
                    self._update_dual_frame(raw_seq[t], refined_seq[t])
                    time.sleep(1 / self.fps)
        except KeyboardInterrupt:
            print("🛑 Stopped.")
