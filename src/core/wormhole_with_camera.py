from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.utils.coordinates import z_value
from src.core.wormhole import WormHole


class WormHoleWithCamera(WormHole):
    """
    Extends the WormHole class by adding a virtual camera to visualize
    particle trajectories and check intersections with a camera plane.
    """

    def __init__(self, r_max: float, b: float, num_points: int = 100) -> None:
        """
        Initializes the wormhole with a virtual camera.

        Args:
            r_max (float): Maximum radial distance for the wormhole.
            b (float): Wormhole throat parameter.
            num_points (int): Resolution of the wormhole surface mesh.
        """
        super().__init__(r_max, b, num_points)
        self.camera_angle: float = 0.0  # Angle of the camera around the wormhole
        self.camera_height: float = 2.0  # Height above the wormhole surface
        self.camera_size: list[float] = [3.5, 3.5]  # [width, height] of the camera plane
        self.intersection_count: int = 0

    def update_camera_position(self, angle: float) -> None:
        """
        Updates the camera angle along the circular rim and checks for intersections.

        Args:
            angle (float): New camera angle in radians.
        """
        self.camera_angle = angle
        self.check_intersections()

    def get_camera_transform(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the camera's position and orientation in 3D space.

        Returns:
            tuple: (position, normal, right_vector, up_vector)
        """
        r = self.r_max
        x = r * np.cos(self.camera_angle)
        y = r * np.sin(self.camera_angle)
        z = z_value(np.array([r]), self.b)[0] + self.camera_height

        # Compute normal to the surface
        dz_dr = self.b / np.sqrt(r**2 - self.b**2)
        normal = np.array([-dz_dr * x / r, -dz_dr * y / r, 1.0])
        normal /= np.linalg.norm(normal)

        # Default up vector
        up_vector = np.array([0.0, 0.0, 1.0])
        right_vector = np.cross(normal, up_vector)

        # Rotate the camera view by 90° around its normal
        rotation = R.from_rotvec(normal * np.pi / 2)
        up_vector = rotation.apply(up_vector)
        right_vector = rotation.apply(right_vector)

        return np.array([x, y, z]), normal, right_vector, up_vector

    def check_intersections(self) -> None:
        """
        Checks how many particle trajectories intersect with the camera plane.
        """
        self.intersection_count = 0
        cam_pos, cam_normal, *_ = self.get_camera_transform()

        for i in range(self.num_particles):
            points = np.column_stack((self.x_part[i], self.y_part[i], self.z_part[i]))
            dots = np.dot(points - cam_pos, cam_normal)
            sign_changes = np.where(np.diff(np.sign(dots)))[0]
            if len(sign_changes) > 0:
                self.intersection_count += 1

    def visualize_with_camera(
        self,
        x_part: list[np.ndarray],
        y_part: list[np.ndarray],
        z_part: list[np.ndarray],
        alpha: float = 0.3
    ) -> None:
        """
        Visualizes the wormhole, particle trajectories, and a virtual camera plane.

        Args:
            x_part (list[np.ndarray]): x-coordinates of trajectories.
            y_part (list[np.ndarray]): y-coordinates of trajectories.
            z_part (list[np.ndarray]): z-coordinates of trajectories.
            alpha (float): Transparency level for the wormhole surface.
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Draw wormhole surface (top and mirrored bottom)
        ax.plot_surface(self.X, self.Y, self.Z, color='blue', alpha=alpha)
        ax.plot_surface(self.X, self.Y, -self.Z, color='blue', alpha=alpha)

        # Plot particle trajectories
        colors = plt.cm.tab10.colors
        for i in range(self.num_particles):
            ax.plot(x_part[i], y_part[i], z_part[i], color=colors[i % len(colors)], lw=1.5)

        # Draw camera
        cam_pos, cam_normal, right_vec, up_vec = self.get_camera_transform()
        width, height = self.camera_size

        corners = [
            cam_pos - right_vec * width / 2 - up_vec * height / 2,
            cam_pos + right_vec * width / 2 - up_vec * height / 2,
            cam_pos + right_vec * width / 2 + up_vec * height / 2,
            cam_pos - right_vec * width / 2 + up_vec * height / 2
        ]

        verts = [corners]
        camera = Poly3DCollection(verts, alpha=0.7, color='red', edgecolor='black')
        ax.add_collection3d(camera)

        # Display intersection count
        ax.set_title(f"Intersecting trajectories: {self.intersection_count}", fontsize=14)

        # Set axis limits
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_zlim(-3.5, 3.5)

        # Camera orbit animation
        def update(frame: int):
            ax.view_init(elev=10 + 20 * np.sin(np.radians(frame)), azim=frame)
            return ax

        ani = FuncAnimation(fig, func=update, frames=np.arange(0, 360, 5), interval=50, blit=False)

        plt.tight_layout()
        plt.show()
