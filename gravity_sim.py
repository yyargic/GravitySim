import numpy as np
import pygame

class Particle:
    """Represents a particle."""

    def __init__(self, mass, position_x, position_y, velocity_x, velocity_y):
        """Initializes a particle by its mass, position, and velocity."""
        self.mass = mass
        self.position = np.array([float(position_x), float(position_y)])
        self.velocity = np.array([float(velocity_x), float(velocity_y)])
        self.acceleration = np.array([0.0, 0.0])

    def __str__(self):
        """Prints the attributes of a particle."""
        s = f"""
            Particle
            mass    : {self.mass}
            pos.    : {self.position[0]}, {self.position[1]}
            vel.    : {self.velocity[0]}, {self.velocity[1]}                       
            """
        return s

    __repr__ = __str__

class State:
    """Represents a state consisting of multiple particles."""

    def __init__(self, NewtonG, particles, dt):
        """Initializes a state."""
        self.NewtonG = NewtonG
        self.particles = particles
        self.dt = dt

        self.number_of_particles = len(particles)
        self.masses = np.array([particle.mass for particle in particles])
        self.positions = np.vstack(
            [particle.position for particle in particles])
        self.velocities = np.vstack(
            [particle.velocity for particle in particles])

        self.accelerations = self.calculate_acceleration()
        self.momentum = self.calculate_momentum()
        self.kinetic_energy = self.calculate_kinetic()
        self.potential_energy = self.calculate_potential()
        self.energy = self.calculate_energy()
        self.center_of_mass = self.calculate_center_of_mass()
        self.angular_momentum = self.calculate_angular_momentum()

    def __str__(self):
        """Prints the attributes of a state."""
        s = f"""
            State of {self.number_of_particles} particles
            energy   :  {np.round(self.energy,3)}  =  {np.round(self.kinetic_energy,3)} (kin.) + {np.round(self.potential_energy,3)} (pot.)
            momentum :  [{np.round(self.momentum[0],3)}, {np.round(self.momentum[1],3)}],  (norm: {np.round(np.linalg.norm(self.momentum),3)})
            ang.mom. :  {np.round(self.angular_momentum,3)}
            """
        return s

    __repr__ = __str__

    def calculate_acceleration(self):
        """Calculates the acceleration of particles from Newton's gravitational force."""
        dist = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        result = np.linalg.norm(dist, axis=-1)
        result += np.eye(self.number_of_particles)
        result = 1 / result ** 3
        result -= np.eye(self.number_of_particles)
        result[result == np.inf] = 0
        result = result[:, :, np.newaxis] * dist
        result = self.NewtonG * self.masses[:, np.newaxis] * result
        result = np.sum(result, axis=1)
        return result

    def calculate_potential(self):
        """Calculates the total potential energy in the state."""
        result = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        result = np.linalg.norm(result, axis=-1)
        result += np.eye(self.number_of_particles)
        result = 1 / result
        result -= np.eye(self.number_of_particles)
        result[result == np.inf] = 0
        result = - self.NewtonG * self.masses[:, np.newaxis] * self.masses[np.newaxis, :] * result
        result = np.sum(result)
        return result

    def calculate_kinetic(self):
        """Calculates the total kinetic energy in the state."""
        return 0.5 * np.linalg.norm(self.velocities, axis=1) ** 2 @ self.masses

    def calculate_energy(self):
        """Calculates the total energy in the state."""
        return self.kinetic_energy + self.potential_energy

    def calculate_momentum(self):
        """Calculates the total momentum in the state."""
        return self.masses @ self.velocities

    def calculate_center_of_mass(self):
        """Calculates the center of mass of the state."""
        return self.masses @ self.positions / np.sum(self.masses)

    def calculate_angular_momentum(self):
        """Calculates the total angular momentum of the state w.r.t. its center of mass."""
        result = self.masses[:, np.newaxis] * (self.velocities - self.momentum / np.sum(self.masses))
        result = np.cross(self.positions - self.center_of_mass, result)
        return - np.sum(result)

    def update(self):
        """Calculates the attributes of particles in each time step using Verlet integration."""
        x0 = self.positions
        v0 = self.velocities
        a0 = self.accelerations
        dt = self.dt
        self.positions = x0 + v0 * dt + 0.5 * a0 * dt * dt
        a1 = self.accelerations = self.calculate_acceleration()
        self.velocities = v0 + 0.5 * (a0 + a1) * dt

    def update_trackables(self):
        """Updates the remaining attributes of the particles and the state."""
        for i, particle in enumerate(self.particles):
            particle.position = self.positions[i]
            particle.velocity = self.velocities[i]
            particle.acceleration = self.accelerations[i]
        self.momentum = self.calculate_momentum()
        self.kinetic_energy = self.calculate_kinetic()
        self.potential_energy = self.calculate_potential()
        self.energy = self.calculate_energy()
        self.center_of_mass = self.calculate_center_of_mass()

class Simulation:
    """Executes a pygame simulation for the gravitational dynamics of the particles."""

    def __init__(self, particles, width=1000, height=500, refresh_rate=100, dt=0.005, NewtonG=1):
        """Initializes the simulation."""
        pygame.init()
        self.screen = pygame.display.set_mode([width, height])
        self.font = pygame.font.SysFont(None, 15)
        self.refresh_rate = refresh_rate
        self.particles = particles
        self.colors = [(np.random.randint(0, 255),
                        np.random.randint(100, 255),
                        np.random.randint(0, 245), 100)
                       for i in range(len(particles))]
        self.state = State(NewtonG, self.particles, dt)
        self.running = True
        self.pause = False
        self.width = width
        self.height = height
        self.trajectory = False
        self.position_display = False

        self.run()

    def get_key_events(self):
        """Manages what happens when the user clicks certain keyboard buttons."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                if event.key == pygame.K_q:
                    self.running = False
                if event.key == pygame.K_t:
                    self.screen.fill((0, 0, 0))
                    if self.trajectory == False and self.position_display:
                        self.position_display = False
                    self.trajectory = not self.trajectory
                if event.key == pygame.K_p:
                    if self.position_display == False and self.trajectory:
                        self.trajectory = False
                    self.position_display = not self.position_display

    def draw_text(self):
        """Displays a text on the top-left corner with the state attributes."""
        txt = self.font.render(
            f"Energy: {np.round(self.state.energy, 3)}, Kinetic: {np.round(self.state.kinetic_energy, 3)}, Potential: {np.round(self.state.potential_energy, 3)}",
            True, (255, 255, 255))
        self.screen.blit(txt, (0, 0))
        txt = self.font.render(
            f"Momentum: [{np.round(self.state.momentum[0],3)}, {np.round(self.state.momentum[1],3)}],  (norm: {np.round(np.linalg.norm(self.state.momentum),3)})",
            True, (255, 255, 255))
        self.screen.blit(txt, (0, 10))
        txt = self.font.render(
            f"Ang. momentum: {np.round(self.state.angular_momentum, 3)}",
            True, (255, 255, 255))
        self.screen.blit(txt, (0, 20))

    def draw_particle(self):
        """Draws the particles and displays their attributes."""
        for i, particle in enumerate(self.particles):
            if self.position_display:
                txt = self.font.render(
                    f"Particle {i+1}  | mass: {np.round(particle.mass)} pos: {np.round(particle.position, 2)} vel: {np.round(particle.velocity, 2)}",
                    True, self.colors[i])
                self.screen.blit(txt, (0, 50 + 10 * i))
            pygame.draw.circle(self.screen,
                               self.colors[i],
                               list(particle.position - self.state.center_of_mass
                                    + np.array([self.width / 2, self.height / 2])),
                               np.sqrt(particle.mass))

    def run(self):
        """Runs the simulation."""
        while self.running:
            self.get_key_events()

            if not self.trajectory:
                self.screen.fill((0, 0, 0))
            else:
                pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, 300, 30), width=0)

            self.draw_particle()
            self.draw_text()

            pygame.display.flip()  # Updates what is shown on the screen

            if self.pause == False:
                for i in range(self.refresh_rate):
                    self.state.update()
                self.state.update_trackables()
