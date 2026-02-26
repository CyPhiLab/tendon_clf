import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory():
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.pi/4
    L = 0.5
    r = L/8
    x1 = L/3 * np.cos(theta)
    y1 = r * np.sin(theta) - (L-r)
    x = x1 * np.cos(phi) - y1 * np.sin(phi)
    y = x1 * np.sin(phi) + y1 * np.cos(phi)
    x0 = L * np.cos(theta)
    y0 = L * np.sin(theta)
    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')  # <-- enforce equal scaling
    plt.grid()
    plt.plot(x, y, label='Ellipse')
    plt.plot(x0, y0, label='Circle')
    plt.plot(0, 0, 'ro', label='Origin')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('3D Trajectory')
    # plt.legend()
    plt.show()
    plt.show()


if __name__ == "__main__":
    plot_trajectory()