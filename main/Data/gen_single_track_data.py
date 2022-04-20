import numpy as np
import matplotlib.pyplot as plt
import random
import math

def rk4(t0, x0, f, delta_t, steps):
    """ Simple runge kutta 4 integration method
        integrates d/dt x(t) = f(t, x) starting from (t0, x0) steps times with delta_t stepsize
    """
    trajectory = np.zeros((len(x0) + 1, steps + 1))
    trajectory[:, 0] = [t0, *x0]
    h = delta_t
    t = t0
    x = x0
    for i in range(steps):
        k1 = h * (f(t, x))
        k2 = h * (f(t + h/2, x + k1/2))
        k3 = h * (f(t + h/2, x + k2/2))
        k4 = h * (f(t + h, x + k3))
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        x = x + k
        t = t + h
        trajectory[:, i+1] = [t, *x]
    return trajectory

class RHS():
    def __init__(self,  decay_p = 0.5,  
                        growth_p = 0.5,
                        acceleration_range = [-4, 4],
                        steering_angle_range = [-22/28, 22/28], #~45° max
                        frequency_scales = 5,
                        frequency_range = [-1, 1],
                        wheelbase = 2.78): 
        self.acceleration_range = acceleration_range            
        self.steering_angle_range = steering_angle_range
        self.frequency_scales = frequency_scales
        self.frequency_range = frequency_range
        self.wheelbase = wheelbase
        self.acceleration_width = self.acceleration_range[1] - self.acceleration_range[0]
        self.base_acceleration = self.acceleration_range[0] + self.acceleration_width*random.random()
        self.steering_width = self.steering_angle_range[1] - self.steering_angle_range[0]
        self.base_steering = self.steering_angle_range[0] + self.steering_width*random.random()
        
        
        # self.use_linear_decay = random.random() < decay_p
        # self.use_linear_growth = False if self.use_linear_decay else random.random() < growth_p

        # linear_decay = random.random()
        # self.linear_decay = lambda t: 1 - t*linear_decay
        # self.linear_growth = lambda t: t*linear_decay
        self.frequencies_steering = self.get_random_frequencies_and_amplitudes()
        self.frequencies_acceleration = self.get_random_frequencies_and_amplitudes()
        
    
    def get_random_frequencies_and_amplitudes(self):
        frequencies = list()
        for i in range(self.frequency_scales):
            frequency_width = (self.frequency_range[1] - self.frequency_range[0])
            sign_s = random.randint(-1, 1)
            sign_c = random.randint(-1, 1)
            sign_a = random.randint(-1, 1)
            omega_s = sign_s*2**i + self.frequency_range[0] + frequency_width*random.random()
            omega_c = sign_c*2**i + self.frequency_range[0] + frequency_width*random.random()
            amplitude = sign_a*(2*random.random() - 1)/(2**i)
            frequencies.append((amplitude, omega_s, omega_c))
        return frequencies

    def get_steering(self, t):
        
        steering = self.base_steering
        for (amplitude, omega_s, omega_c) in self.frequencies_steering:
            steering += self.steering_width*amplitude*(math.cos(omega_c*t) + math.sin(omega_s*t))
        steering = np.clip(steering, self.steering_angle_range[0], self.steering_angle_range[1])
        return steering

    def get_acceleration(self, t):
        acceleration = self.base_acceleration
        
        for (amplitude, omega_s, omega_c) in self.frequencies_acceleration:
            acceleration += self.acceleration_width*amplitude*(math.cos(omega_c*t) + math.sin(omega_s*t))
        acceleration = np.clip(acceleration, self.acceleration_range[0], self.acceleration_range[1])
        return acceleration

    def plot_controls(self, ax, t_range = [0, 1], steps = 100):
        T = np.linspace(t_range[0], t_range[1], steps)
        steering, acc = np.zeros(steps), np.zeros(steps)
        for i, t in enumerate(T):
            steering[i] = self.get_steering(t)
            acc[i] = self.get_acceleration(t)
        ax.plot(T, steering, label="steering")
        ax.plot(T, acc, label="acceleration")
        ax.legend()

    
    def __call__(self, t, X):
        (x, y, v, yaw) = X
        xdot = v*math.cos(yaw)
        ydot = v*math.sin(yaw)
        vdot = self.get_acceleration(t)
        yawdot = v/self.wheelbase*math.tan(self.get_steering(t))
        return np.array(xdot, ydot, vdot, yawdot)


def plot_random_controls():
    n = 10
    m = 3
    fig, ax = plt.subplots(m, n, sharey='row')
    for i in range(n):
        for k in range(m):
            example_rhs = RHS()
            example_rhs.plot_controls(ax[k, i])
        
    plt.show()


def test():
    """ Give me a sin
    """
    f = lambda t, x : np.array((x[1], -x[0]))
    steps = 100
    delta_t = 0.1
    x0 = [0, 1]
    t0 = 0
    trajectory = rk4(t0, x0, f, delta_t, steps)
    X = trajectory[0, :]
    Y = trajectory[1, :]
    plt.plot(X, Y)
    plt.show()

if __name__ == "__main__":
    plot_random_controls()
