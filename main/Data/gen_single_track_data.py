import numpy as np
import matplotlib.pyplot as plt
import random
import math

def random_number(range):
    return range[0] + (range[1]-range[0])*random.random()

def rk4(t0, x0, f, delta_t, steps):
    """ Simple runge kutta 4 integration method
        integrates d/dt x(t) = f(t, x) starting from (t0, x0) steps times with delta_t stepsize
    """
    trajectory = np.zeros((len(x0) + 1, steps + 1))
    trajectory[:, 0] = [t0, *x0]
    controls = np.zeros((len(f.get_controls(0)), steps + 1))
    controls[:, 0] = f.get_controls(t0)
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
        controls[:, i+1] = f.get_controls(t)
    return trajectory, controls

class RHS():
    def __init__(self,  decay_p = 0.5,  
                        growth_p = 0.5,
                        acceleration_range = [-4, 4],
                        steering_angle_range = [-22/28, 22/28], #~45° max
                        frequency_scales = 6,
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
            sign_s = random.randint(-1, 1)
            sign_c = random.randint(-1, 1)
            sign_a = random.randint(-1, 1)
            omega_s = sign_s*2**i + random_number(self.frequency_range)
            omega_c = sign_c*2**i + random_number(self.frequency_range)
            amplitude = sign_a*random.random()/(2**i)
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

    def get_controls(self, t):
        return np.array((self.get_acceleration(t), self.get_steering(t)))

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
        return np.array((xdot, ydot, vdot, yawdot))


def plot_random_controls():
    n = 10
    m = 3
    fig, ax = plt.subplots(m, n, sharey='row')
    for i in range(n):
        for k in range(m):
            example_rhs = RHS()
            example_rhs.plot_controls(ax[k, i])
        
    plt.show()

def generate_random_trajectory( delta_t = 0.01,
                                steps = 1000,
                                initial_xrange = [-100, 100],
                                initial_vrange = [-5, 5],
                                initial_yawrange = [0, 2*math.pi],
                                ):
    x0, y0 = random_number(initial_xrange), random_number(initial_xrange)
    v0 = random_number(initial_vrange)
    yaw0 = random_number(initial_yawrange)
    X0 = np.array((x0, y0, v0, yaw0))
    t0 = 0
    f = RHS()
    trajectory, controls = rk4(t0, X0, f, delta_t, steps)

    return trajectory, controls

def save_trajectory(trajectory, controls, index, save_path = "/generated_data"):
    import os
    os.makedirs(save_path, exist_ok=True)

                
    with open(f"{save_path}/{index:05d}.csv", 'w+') as f:
        f.write(f"time,velocity,orientation,acceleration,steering wheel angle,")
        for (T, C) in zip(trajectory.T, controls.T):
            #print(T)
            (t, x, y, v, yaw) = T
            (acceleration, steering) = C
            f.write(f"\n{t},{v},{yaw},{acceleration},{steering}")

def generate_dataset(n = 1000, save_path = "generated_data/"):
    from tqdm import tqdm
    pbar = tqdm(total = n)
    for i in range(n):
        trajectory, controls = generate_random_trajectory()
        save_trajectory(trajectory, controls, i, save_path)
        pbar.update()


def plot_random_trajectory():
    m = 4
    fig, ax = plt.subplots(m, 4)
    for k in range(m):
        
        trajectory, controls = generate_random_trajectory()
        
        X = trajectory[1, :]
        Y = trajectory[2, :]
        T = trajectory[0, :]
        V = trajectory[3, :]
        acceleration = controls[0, :]
        steering = controls[1, :]
        ax[k, 0].plot(X, Y, label = 'trajectory', color='black')
        ax[k, 0].plot(X[:1], Y[:1], marker = 'o', color='red', label = 'initial position')
        ax[k, 0].set_aspect('equal', 'box')
        ax[k, 0].set(xlabel="x [m]", ylabel="y[m]")
        ax[k, 1].plot(T, V, label = 'speed')#
        ax[k, 1].set(xlabel="t [s]", ylabel=r"v [m$s^{-1}$]")
        ax[k, 2].plot(T, acceleration, label = 'acceleration', color='red')
        ax[k, 2].set(xlabel="t [s]", ylabel=r"a [m$s^{-2}$]")
        ax[k, 3].plot(T, steering, label = 'steering', color='red')
        ax[k, 3].set(xlabel="t [s]", ylabel=r'$\delta$ [rad]')
        for a in ax[k, :]:
            a.legend()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plt.tight_layout()
    plt.show()



def test_rk():
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
    generate_dataset()
