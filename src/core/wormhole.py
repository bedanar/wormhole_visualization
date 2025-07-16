import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj, ellipk
from src.utils.coordinates import z_value, z_value_for_singularity


class WormHole:
    """
    Creates a wormhole geometry in Cartesian coordinates, 
    simulating trajectories over a curved surface defined by parameter `b`.
    """
    def __init__(self, 
            r_max: float,
            b: float,
            num_points: int = 100):
        """
        Initializes the WormHole object with given maximum radius, parameter b, and resolution.

        Args:
            r_max (float):    Maximum radial distance.
            b (float):        Wormhole throat parameter.
            num_points (int): Number of points for mesh grid resolution.
        """

        self.b = b
        self.r_max = r_max 

        # r and phi initialization
        r = np.linspace(b, self.r_max, num_points)
        phi = np.linspace(0, 2*np.pi, num_points)

        # createing a meshgrid in cylindrical coordinates
        R, Phi = np.meshgrid(r, phi)
        
        # surface coordinates in cartesian coordinates
        self.X = R * np.cos(Phi)
        self.Y = R * np.sin(Phi)
        self.Z = z_value(R, b)


    def calculate_regular_trajectory(self, rt: list[float], num_points: int) -> tuple[np.array, np.array]:
        """Calculates a regular geodesic trajectory over the wormhole surface.

        Args:
            rt (float):                    Turning point radius.
            num_points (int):              Number of trajectory points.

        Returns:
            tuple[np.ndarray, np.ndarray]: (phi, r values)
        """
        m = self.b**2 / rt**2
        phi_max = ellipk(m)
        phi = np.linspace(-phi_max, phi_max, num_points)
        r_vals = []
        for p in phi:
            sn, cn, dn, ph = ellipj(p, m)
            r = rt * dn / cn  # dc = dn/cn
            r_vals.append(r)
        
        return phi, np.array(r_vals)

    def calculate_singular_trajectory(self, rt: list[float], num_points: int) -> tuple[np.array, np.array]:
        """
        Calculates a singular geodesic trajectory (trajectories that reach the throat).

        Args:
            rt (float):                    Turning point radius.
            num_points (int):              Number of trajectory points.

        Returns:
            tuple[np.ndarray, np.ndarray]: (phi, r values)
        """

        m = rt**2 / self.b**2
        phi_max = (rt / self.b) * ellipk(m)
        phi = np.linspace(-phi_max, phi_max, num_points)
        
        r_vals = []
        for p in phi:
            u = self.b * p / rt
            sn, cn, dn, ph = ellipj(u, m)
            r = self.b * dn / cn  # dc = dn/cn
            r_vals.append(r)
        
        return phi, np.array(r_vals)
    
    def projection(self, num_particles: int, flag: str) -> tuple:  
        """
        Projects particle trajectories over the wormhole surface.

        Args:
            num_particles (int): Number of geodesic trajectories to generate.
            flag (str):          Type of geodesic ('regular' or 'singular').

        Returns:
            tuple:               (rt values, x coordinates, y coordinates, z coordinates)
        """

        x_part, y_part, z_part = [], [], []
        self.num_particles = num_particles

        if flag == 'regular':
            rt = [random.uniform(1.0, self.r_max) for _ in range(num_particles)]
        elif flag == 'singular':
            rt = [random.uniform(0.1, 0.999) for _ in range(num_particles)]
        else:
            raise ValueError("Flag must be 'regular' or 'singular'.")

        # Ensure uniqueness of rt values
        while len(set(rt)) != len(rt):
            if flag == 'regular':
                rt = [random.uniform(1.0, self.r_max) for _ in range(num_particles)]
            else:
                rt = [random.uniform(0.1, 0.999) for _ in range(num_particles)]

        for r in rt:
            if flag == 'regular':
                phi, r_vals = self.calculate_regular_trajectory(r, num_points=500)
                z = z_value(r_vals, self.b)
            else:
                phi, r_vals = self.calculate_singular_trajectory(r, num_points=500)
                z = z_value_for_singularity(r_vals, self.b)

            x = r_vals * np.cos(phi)
            y = r_vals * np.sin(phi)

            x_part.append(x)
            y_part.append(y)
            z_part.append(z)

        return rt, x_part, y_part, z_part
            
    def visualize(
            self,
            rt: list[float],
            x_part: list[np.ndarray],
            y_part: list[np.ndarray],
            z_part: list[np.ndarray],
            alpha: float = 0.1,
            static_view: bool = False
        ) -> None:
        """
        Visualizes the wormhole surface and the particle trajectories.

        Args:
            rt (list[float]):          List of turning point radii for each trajectory.
            x_part (list[np.ndarray]): x-coordinates of the trajectories.
            y_part (list[np.ndarray]): y-coordinates of the trajectories.
            z_part (list[np.ndarray]): z-coordinates of the trajectories.
            alpha (float):             Transparency level for the surface.
            static_view (bool):        Whether to use a static view (no rotation).
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the wormhole surface (top and mirrored bottom)
        ax.plot_surface(self.X, self.Y, self.Z, color='blue', alpha=alpha, edgecolor='black', rstride=1, cstride=1)
        ax.plot_surface(self.X, self.Y, -self.Z, color='blue', alpha=alpha, edgecolor='black', rstride=1, cstride=1)

        colors = plt.cm.tab10.colors

        for i in range(self.num_particles):
            # Plot only points within a bounding box (cropping)
            x_masked = x_part[i]
            y_masked = y_part[i]
            z_masked = z_part[i]

            mask = (
                (-2.0 <= x_masked) & (x_masked <= 2.0) &
                (-1.5 <= y_masked) & (y_masked <= 1.5) &
                (0.0 <= z_masked) & (z_masked <= 3.0)
            )

            ax.plot(
                x_masked[mask], y_masked[mask], z_masked[mask],
                color=colors[i % len(colors)], lw=2.5,
                label=f'$r_t$={rt[i]:.2f}'
            )

        ax.set_xlim(-self.r_max, self.r_max)
        ax.set_ylim(-self.r_max, self.r_max)
        ax.set_zlim(-self.r_max, self.r_max)
        
        if static_view:
            ax.view_init(elev=90, azim=0)

        plt.tight_layout()
        plt.show()
