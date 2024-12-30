from python_ugv_sim.utils import vehicles, environment
import numpy as np
import pygame
import matplotlib.pyplot as plt

if __name__=='__main__':
    # Initialize pygame
    pygame.init()

    # Initialize robot and time step
    x_init = np.array([1,1,np.pi/2])
    robot = vehicles.DifferentialDrive(x_init)
    dt = 0.01

    path_save = []
    path_m_save = []

    # Initialize and display environment
    env = environment.Environment(map_image_path="./python_ugv_sim/maps/map_blank.png")

    running = True
    u = np.array([0.,0.]) # Controls
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running = False
            u = robot.update_u(u,event) if event.type==pygame.KEYUP or event.type==pygame.KEYDOWN else u

        rx, ry, _ = robot.x
        rx_pix, ry_pix = env.position2pixel((rx,ry))
        path_save.append((rx_pix, ry_pix))
        
        rx_m, ry_m = env.pixel2position((rx_pix, ry_pix))
        path_m_save.append((rx_m, ry_m))
        
        robot.move_step(u,dt)
        env.show_map()
        env.show_robot(robot)
        pygame.display.update()

    # Save paths to files
    # Save pixel coordinates
    with open('robot_path_pixels.txt', 'w') as f:
        f.write("x_pixel,y_pixel\n")  # Header
        for point in path_save:
            f.write(f"{point[0]},{point[1]}\n")

    # Save meter coordinates
    with open('robot_path_meters.txt', 'w') as f:
        f.write("x_meters,y_meters\n")  # Header
        for point in path_m_save:
            f.write(f"{point[0]},{point[1]}\n")

    # Create plots
    # Plot in pixels
    path_array = np.array(path_save)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(path_array[:, 0], path_array[:, 1], 'b-', label='Robot Path (pixels)')
    plt.title('Robot Path Trajectory (Pixels)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.grid(False)
    plt.legend()

    # Plot in meters
    path_m_array = np.array(path_m_save)
    plt.subplot(1, 2, 2)
    plt.plot(path_m_array[:, 0], path_m_array[:, 1], 'r-', label='Robot Path (meters)')
    plt.title('Robot Path Trajectory (Meters)')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.grid(False)
    plt.legend()

    plt.tight_layout()
    plt.savefig('robot_path_plots.png')
    plt.show()

    pygame.quit()