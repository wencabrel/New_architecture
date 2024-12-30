from python_ugv_sim.utils import vehicles, environment
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__=='__main__':
    # Initialize pygame
    pygame.init()

    # Initialize matplotlib figure
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    line1, = ax1.plot([], [], 'b-', label='Robot Path (pixels)')
    line2, = ax2.plot([], [], 'r-', label='Robot Path (meters)')
    
    # Setup the plots
    ax1.set_title('Robot Path Trajectory (Pixels)')
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_title('Robot Path Trajectory (Meters)')
    ax2.set_xlabel('X Position (meters)')
    ax2.set_ylabel('Y Position (meters)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()

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
    
    # Counter for plot updates
    plot_update_counter = 0
    plot_update_frequency = 20  # Reduced update frequency
    
    clock = pygame.time.Clock()  # Add clock for consistent frame rate

    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                u = robot.update_u(u, event)
        
        # Move robot
        robot.move_step(u, dt)
        
        # Update positions
        rx, ry, phi = robot.x
        rx_pix, ry_pix = env.position2pixel((rx,ry))
        path_save.append((rx_pix, ry_pix, phi))
        
        rx_m, ry_m = env.pixel2position((rx_pix, ry_pix))
        path_m_save.append((rx_m, ry_m, phi))
        
        # Update pygame display
        env.show_map()
        env.show_robot(robot)
        pygame.display.update()
        
        # Update plots less frequently
        plot_update_counter += 1
        if plot_update_counter >= plot_update_frequency:
            plot_update_counter = 0
            
            try:
                # Update pixel plot
                path_array = np.array(path_save)
                line1.set_data(path_array[:, 0], path_array[:, 1])
                ax1.relim()
                ax1.autoscale_view()
                
                # Update meter plot
                path_m_array = np.array(path_m_save)
                line2.set_data(path_m_array[:, 0], path_m_array[:, 1])
                ax2.relim()
                ax2.autoscale_view()
                
                # Draw updated plots
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except:
                pass  # Ignore any plotting errors to keep robot control smooth
        
        clock.tick(60)  # Maintain 60 FPS

    # Save paths to files
    with open('robot_path_pixels.txt', 'w') as f:
        f.write("x_pixel, y_pixel, heading_rad\\n")
        for point in path_save:
            f.write(f"{point[0]}, {point[1]}, {point[2]}\n")

    with open('robot_path_meters.txt', 'w') as f:
        f.write("x_meters, y_meters, heading_rad\n")
        for point in path_m_save:
            f.write(f"{point[0]}, {point[1]}, {point[2]}\n")

    # Save final plot
    plt.savefig('robot_path_plots.png')
    plt.close()
    pygame.quit()