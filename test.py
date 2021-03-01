from gravity_sim import *

def gas_random(n):
    """Generates a random gas of n particles."""
    return [Particle(np.random.uniform(1., 25.),
                     np.random.uniform(0., 800.),
                     np.random.uniform(0., 300.),
                     np.random.uniform(-2., 2.),
                     np.random.uniform(-2., 2.)
                     ) for i in range(n)]

"""Sample gas with 1 star and 9 comets."""
gas1 = [
    Particle(200, 0, 0, 0, 0.),
    Particle(1, 50, 0, 0, -4),
    Particle(1, 200, 0, 0, -1),
    Particle(1, 200, 200, -0.8, 0.1),
    Particle(1, -200, -200, 0.8, -0.1),
    Particle(1, 250, 0, 0,-2.5),
    Particle(2, 250, 400, 0, -0.9),
    Particle(2, 650, 400, 0, 0.9),
    Particle(2, 400, 250, -0.6, 0),
    Particle(2, 400, 650, 0.6, 0)]

"""Sample gas with planets, moons and comets."""
gas2 = [
    Particle(250, 0, 0, 0, 0),
    Particle(25,  410,   0,    0,  1.2),
    Particle(1,   410,  75, -0.3,  1.2),
    Particle(21, -410,   0,    0, -1.1),
    Particle(1,  -420, -50,  0.4,   -1),
    Particle(1, 0,   75, -1.5, 0),
    Particle(1, 0,  -75,  1.5, 0),
    Particle(1, 0,  110, -1.2, 0),
    Particle(1, 0, -110,  1.2, 0)]

"""Sample gas with a simple, symmetric configuration."""
gas3 = [
    Particle(100, 500, 250,  0, 0),
    Particle(1,   500, 450,  1, 0),
    Particle(1,   500,  50, -1, 0)]

"""Sample gas with symmetric double stars with planets. Best with NewtonG=10."""
gas4 = [
    Particle(100,  200, 0, 0, 1),
    Particle(4,  200, 50, -4, 1),
    Particle(100, -200, 0, 0, -1),
    Particle(4,  -200, -50, 4, -1),
]

#gas = gas_random(50)
#gas = gas1
#gas = gas2
#gas = gas3
gas = gas4

s = State(1, gas, 0.1)
g = Simulation(gas, dt=0.005, NewtonG=10, width=1200, height=700, refresh_rate=100)
