from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def plot_cartesian_3d(coordinates, plot_type="point", color="blue", label=None):
    """
    Plots a list of (x, y, z) coordinates as points or vectors in a 3D space.

    Parameters:
    - coordinates: List of tuples/lists, e.g., [(1, 2, 3), (-3, 4, 2)]
    - plot_type: 'point' to scatter plot, or 'vector' to draw arrows from origin
    - color: Color of the points or vectors
    - label: Legend label for the data series
    """
    # 1. Create the figure and add a 3D subplot
    fig = plt.subplots(figsize=(8, 8))
    ax = plt.subplot(111, projection="3d")

    # 2. Extract X, Y, and Z arrays
    coords = np.array(coordinates)
    X = coords[:, 0]
    Y = coords[:, 1]
    Z = coords[:, 2]

    # 3. Plot based on user preference
    if plot_type == "point":
        ax.scatter(X, Y, Z, color=color, s=50, zorder=5, label=label)

    elif plot_type == "vector":
        # quiver inputs: X_origin, Y_origin, Z_origin, U_dir, V_dir, W_dir
        # np.zeros_like makes all arrows start at the origin (0,0,0)
        ax.quiver(
            np.zeros_like(X),
            np.zeros_like(Y),
            np.zeros_like(Z),
            X,
            Y,
            Z,
            color=color,
            arrow_length_ratio=0.1,
            label=label,
        )
    else:
        raise ValueError("plot_type must be either 'point' or 'vector'")

    # 4. Draw reference axis lines intersecting at (0,0,0)
    max_val = max(max(abs(X)), max(abs(Y)), max(abs(Z)), 1)
    axis_line = np.linspace(-max_val * 1.1, max_val * 1.1, 2)

    ax.plot(
        axis_line, [0, 0], [0, 0], color="black", alpha=0.4, linestyle=":"
    )  # X axis
    ax.plot(
        [0, 0], axis_line, [0, 0], color="black", alpha=0.4, linestyle=":"
    )  # Y axis
    ax.plot(
        [0, 0], [0, 0], axis_line, color="black", alpha=0.4, linestyle=":"
    )  # Z axis

    # 5. Dynamically calculate padding for symmetric limits
    padding = max_val * 1.2
    ax.set_xlim(-padding, padding)
    ax.set_ylim(-padding, padding)
    ax.set_zlim(-padding, padding)

    # 6. Axis labels
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    if label:
        ax.legend()

    return fig, ax


def plot_cartesian_2d(
    data: list,
    plot_type: Literal["point", "vector"],
):
    """Plots 2D Cartesian coordinates as points or directional vectors from the origin.

    Parameters:
    - data: List of tuples in the format (x, y)
    - plot_type: Choose "point" for markers or "vector" for directional arrows
    """
    # 1. Setup standard 2D linear figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # 2. Extract X and Y coordinates
    x_coords = [item[0] for item in data]
    y_coords = [item[1] for item in data]

    # 3. Route plot based on the literal value
    if plot_type == "point":
        ax.scatter(
            x_coords,
            y_coords,
            color="crimson",
            edgecolor="black",
            s=80,
            zorder=3,
            label="Points",
        )

    elif plot_type == "vector":
        for x, y in zip(x_coords, y_coords):
            # Use quiver for vectors to render clean mathematical arrows
            ax.quiver(
                0,
                0,
                x,
                y,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.007,
                label=f"v({x:.1f}, {y:.1f})",
            )

    # 4. Standard styling configurations
    # Draw crosshairs for origin axes (0,0)
    ax.axhline(0, color="black", linewidth=1.2, zorder=1)
    ax.axvline(0, color="black", linewidth=1.2, zorder=1)
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)

    # Set dynamic, symmetric padding so vectors/points aren't cut off at the border
    max_val = max(max(abs(x) for x in x_coords), max(abs(y) for y in y_coords), 1)
    padding = max_val * 1.2
    ax.set_xlim(-padding, padding)
    ax.set_ylim(-padding, padding)

    # Enforce a perfectly square 1:1 aspect ratio for geometric accuracy
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    plt.title(
        f"2D Cartesian Mapping ({plot_type.capitalize()} Mode)", va="bottom", pad=15
    )

    return fig, ax


def plot_polar_data(
    data: list[tuple[float, float]],
    plot_type: Literal["point", "vector"],
    use_degrees: bool = True,
):
    """
    Plots polar coordinates as points or directional vectors.

    Parameters:
    - data: List of tuples in the format (r, angle)
    - plot_type: Choose "point" for markers or "vector" for lines originating from center
    - use_degrees: Set to True if input angles are in degrees. False if they are in radians.
    """
    # 1. Setup figure with a polar projection
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # 2. Extract radii and angles, then convert to radians if necessary
    radii = [item[0] for item in data]
    angles = [item[1] for item in data]

    if use_degrees:
        angles = np.radians(angles)

    # 3. Route plot based on the literal value
    if plot_type == "point":
        ax.scatter(
            angles,
            radii,
            color="crimson",
            edgecolor="black",
            s=80,
            zorder=3,
            label="Points",
        )

    elif plot_type == "vector":
        for theta, r in zip(angles, radii):
            # Draw line from center (0,0) to target coordinate (theta, r)
            ax.plot(
                [0, theta],
                [0, r],
                linewidth=2.5,
                marker="o",
                label=f"v({r:.1f}, {np.degrees(theta):.0f}°)",
            )

    # 4. Standard styling configurations
    ax.set_theta_zero_location("N")  # Set North (top) as 0 degrees
    ax.set_theta_direction(-1)  # Orient clock-wise for traditional compass bearings
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.title(
        f"Polar Coordinate Mapping ({plot_type.capitalize()} Mode)", va="bottom", pad=15
    )
    return fig, ax


def plot_complex_vectors(numbers):
    """
    Plots a list or array of complex numbers as vectors from the origin.

    Parameters:
    numbers (list or numpy array): Input complex numbers (e.g., [3 + 4j, -2 + 1j])
    """
    # Convert input to a numpy array for easy manipulation
    z = np.atleast_1d(numbers)

    # Extract real (X) and imaginary (Y) components
    x = z.real
    y = z.imag

    # Generate origin coordinates (zeros) for the arrows
    origins = np.zeros_like(x)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot vectors using quiver (X-origin, Y-origin, X-direction, Y-direction)
    # angles='xy', scale_units='xy', and scale=1 ensure arrows match the graph grid perfectly
    ax.quiver(
        origins,
        origins,
        x,
        y,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=["crimson", "royalblue", "forestgreen", "darkorange", "purple"],
        alpha=0.8,
    )

    # Also plot the points at the tips for visual clarity
    ax.scatter(x, y, color="black", zorder=3)

    # Label the coordinates next to the points
    for i, num in enumerate(z):
        ax.text(x[i] + 0.1, y[i] + 0.1, f"{num.real:g} + {num.imag:g}i", fontsize=10)

    # Dynamically scale axis limits based on the largest vector
    max_val = max(max(abs(x)), max(abs(y)), 1) + 1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Configure axes to cross exactly at the center (origin)
    ax.axhline(0, color="black", linewidth=1.2, zorder=1)
    ax.axvline(0, color="black", linewidth=1.2, zorder=1)

    # Add grid, labels, and title
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xlabel("Real Axis ($\Re$)", fontsize=11)
    ax.set_ylabel("Imaginary Axis ($\Im$)", fontsize=11)
    ax.set_title("Complex Plane Vector Plot", fontsize=13, pad=15)
    ax.set_aspect("equal")  # Prevents distortion of angles

    plt.show()
