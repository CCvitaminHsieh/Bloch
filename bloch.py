import numpy as np
import matplotlib.pyplot as plt

MHz = 10**6

# two-level evolution
class Bloch:
    def __init__(self, init_sigma_status: list, qb_param: list,
                 ke: float, tlist: list):
        self.sigma_status = init_sigma_status
        self.qb_param = qb_param
        self.ke = ke
        self.tstart = tlist[0]
        self.tstop = tlist[1]
        self.dt = tlist[2]
        self.t_current = self.tstart
        self.vin = np.array([])

    def _sigma_dot(self, current_sigma_status, t):
        rabi_freq = self.qb_param[0]
        relax = self.qb_param[1]
        decoh = self.qb_param[2] 
        detun = self.qb_param[3]
        #
        current_Vin = self.vin[int(t/self.dt)]
        Omega = rabi_freq * current_Vin
        #
        bloch_eq_matrix = np.array([
            [-1, -detun/decoh, 0.5*np.real(Omega)/decoh],
            [detun_rate/decoh, -1, 0.5*-np.imag(Omega)*0],
            [-2*np.real(Omega)/decoh, 2*np.imag(Omega)*0, -relax/decoh]])
        offset = np.array([0, 0, relax/decoh])
        #
        current_sigma_evo = 2*np.pi*(np.dot(
            bloch_eq_matrix, current_sigma_status) - offset)

        return current_sigma_evo

    def _range_kutta_4(self):
        k1 = self.dt * self._sigma_dot(self.sigma_status,
                                       self.t_current)
        k2 = self.dt * \
            self._sigma_dot(self.sigma_status+0.5*k1,
                            self.t_current+0.5*self.dt)
        k3 = self.dt * \
            self._sigma_dot(self.sigma_status+0.5*k2,
                            self.t_current+0.5*self.dt)
        k4 = self.dt * self._sigma_dot(self.sigma_status+k3, self.t_current)

        self.sigma_status = self.sigma_status + \
            (k1 + 2*k2 + 2*k3 + k4)/6
        self.t_current += self.dt
        return self.sigma_status

    def _setVinArray(self, incident_wave_arr):
        self.vin = incident_wave_arr

    def sigma_evolution(self, incident_wave_arr):
        sigma_collection = list(np.zeros(len(incident_wave_arr)))
        self._setVinArray(incident_wave_arr)
        idx = 0
        while self.t_current < self.tstop:
            # use rk4 to calculate the next iteration of sigma
            sigma_collection[idx] = self._range_kutta_4()
            idx += 1
        return np.array(sigma_collection)


if __name__ == '__main__':
    # sx, sy, sz
    init_sigma_status = [0, 0, -1]
    # qb_param
    rabi_freq =  9*MHz
    relax_rate = 6*MHz
    decoh_rate = 3*MHz
    detun_rate = 0*MHz
    qb_param = [rabi_freq, relax_rate, decoh_rate, detun_rate]
    # atom-field coupling constant
    ke = 7.9 * 10**15
    # time list
    tlist = [0, 1000*10**(-9)*decoh_rate, 1*10**(-9)*decoh_rate]
    xlist = [0, 1000, 1]

    def square_pulse2(time):
        array = []
        for t in range(time[0], time[1], time[2]):
            if t <= 200 or t >= 400:
                array.append(0)
            else:
                array.append(1)
        return array
    vin = square_pulse2(xlist)
    qb_sig_evo = Bloch(init_sigma_status, qb_param, ke, tlist)
    result = qb_sig_evo.sigma_evolution(vin)

    print(result[:, 2])
    plt.plot(result[:, 2])
    plt.show()
