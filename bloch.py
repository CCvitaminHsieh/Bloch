import numpy as np
import matplotlib.pyplot as plt

MHz = 10**6

# two-level evolution
class Bloch:
    def __init__(
            self,
            t: list,
            qb_param: list
            ):
        # initial state is at ground state
        self.sigma_status = [0, 0, -1]
        # qubit properties
        self.rabi_freq = qb_param[0]
        self.relax = qb_param[1]
        self.decoh = qb_param[2] 
        self.detun = qb_param[3]
        # set the initial config of time-axis
        self.tstart, self.tstop, self.dt = t[0], t[1], t[2]
        self.t_current = self.tstart
        # initial incident light
        self.vin = None
        self.sigma_x = np.array([])
        self.sigma_y = np.array([])
        self.sigma_z = np.array([])
        
    def evolution(self):
        self.evo_result = {
            'sigma_x': self.sigma_x,
            'sigma_y': self.sigma_y,
            'sigma_z': self.sigma_z,
            'pe': (1 + self.sigma_z) / 2
            }
        if not isinstance(self.vin, np.ndarray):
            raise TypeError(
                    'Lack of the information about incident light '+\
                    'due to the invalid value "self.vin=None".')
        else:
            sigma_items = list(np.zeros(len(self.vin)))
            idx = 0
            # use runge-kutta-4 to calculate the next iteration of sigma
            while self.t_current < self.tstop:
                sigma_items[idx] = self._rk4()
                idx += 1
            sigma_items = np.array(sigma_items)
            for idx, quant in enumerate(('x', 'y', 'z')):
                self.evo_result[f'sigma_{quant}'] = sigma_items[:, idx]

    def _rk4(self):
        # refer from: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        k1 = self.dt * self._bloch_eq(
            self.sigma_status, self.t_current)
        k2 = self.dt * self._bloch_eq(
            self.sigma_status+0.5*k1, self.t_current+0.5*self.dt)
        k3 = self.dt * self._bloch_eq(
            self.sigma_status+0.5*k2, self.t_current+0.5*self.dt)
        k4 = self.dt * self._bloch_eq(
            self.sigma_status+k3, self.t_current)

        self.sigma_status = self.sigma_status + \
            (k1 + 2*k2 + 2*k3 + k4)/6
        self.t_current += self.dt
        return self.sigma_status
    
    def _bloch_eq(
            self,
            current_sigma_status,
            t):
        Omega = self.rabi_freq * self.vin[int(t/self.dt)]
        bloch_eq_matrix = np.array([
            [-1, -self.detun/self.decoh, 0.5 * np.real(Omega) / self.decoh],
            [self.detun/self.decoh, -1, 0.5*-np.imag(Omega)*0],
            [-2*np.real(Omega)/self.decoh, 2*np.imag(Omega)*0, -self.relax/self.decoh]])
        offset = np.array([0, 0, self.relax/self.decoh])
        # optical bloch equation in two-level system
        next_sigma_items = 2 * np.pi * (
            np.dot(bloch_eq_matrix, current_sigma_status) - offset)
        return next_sigma_items
    
    def plot(self, tag = 'sigma_z'):
        if tag in self.evo_result.keys():
            plt.plot(
                np.linspace(
                    self.tstart,
                    self.tstop,
                    int(abs(self.tstop - self.tstart) / self.dt)),
                self.evo_result[tag])
            plt.show()
        else:
            raise KeyError (f'"{tag}" is not found.')


if __name__ == '__main__':
    # generate a pulse
    def square_pulse(t_array, offset):
        y = np.zeros(
            int((t_array[1] - t_array[0]) / t_array[2]))
        for idx, t in enumerate(
                np.linspace(t_array[0], t_array[1], len(y))):
            y[idx] = 0 if (t <= offset[0] or t >= offset[1]) else 1
        return y

    # qb_param
    rabi_freq =  18 * MHz
    relax_rate = 6 * MHz
    decoh_rate = 3 * MHz
    detun_rate = 0 * MHz
    qb_param = [rabi_freq, relax_rate, decoh_rate, detun_rate]
    
    # create time properties
    t = np.array([0, 1000, 1]) *10**(-9)*decoh_rate
    square_pulse_offset = np.array([200, 400]) *10**(-9)*decoh_rate

    # simulation
    qb = Bloch(t, qb_param)
    qb.vin = square_pulse(t, square_pulse_offset)
    qb.evolution()
    
    # plot evolution
    qb.plot('sigma_z')
