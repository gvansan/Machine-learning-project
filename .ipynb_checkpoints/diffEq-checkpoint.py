import matplotlib.pyplot as plt
import numpy as np
import math

class odes:
    def __init__(self, odes_list, init_vals_list):
        self.odes_list = odes_list
        self.init_vals_list = init_vals_list

    def solve_Runge_Kutta(self, t_i, t_f, h=0.1, stage=1):
        T = t_f - t_i
        n = len(self.odes_list)
        N = math.floor(T / h)

        t_list = np.zeros(N)
        sol_list = np.zeros((N, n))

        t = t_i
        sol_list[0] = self.init_vals_list
        t_list[0] = t_i

        # Euler's forward method
        if stage == 1:
            for i in range(N - 1):
                for j in range(n):
                    f_j = self.odes_list[j](t, sol_list[i])
                    sol_list[i + 1, j] = sol_list[i, j] + h * f_j

                t += h
                t_list[i + 1] = t

        elif stage == 2:
            for i in range(N - 1):
                k_1 = np.zeros(n)
                
                for j in range(n):
                    k_1[j] = self.odes_list[j](t, sol_list[i])

                for k in range(n):
                    k_2 = self.odes_list[k](t + h/2, sol_list[i] + (h/2)*k_1)
                    sol_list[i + 1, k] = sol_list[i, k] + h*k_2

                t += h
                t_list[i + 1] = t

        elif stage == 4:
            for i in range(N - 1):
                k_13 = [np.zeros(n) for i in range(3)]
                
                for j in range(3):
                    for k in range(n):
                        if j == 0:
                            k_13[j][k] = self.odes_list[k](t, sol_list[i])
                        else:
                            k_13[j][k] = self.odes_list[k](t + h/2, sol_list[i] + (h/2)*k_13[j-1])
                
                for l in range(n):
                    k_4 = self.odes_list[l](t + h, sol_list[i] + h*k_13[2])
                    sol_list[i + 1, l] = sol_list[i, l] + (h/6)*(k_13[0][l] + 2*k_13[1][l] + 2*k_13[2][l] + k_4)

                t += h
                t_list[i + 1] = t

        else:
            raise ValueError("Stage not implemeted.")

        return t_list, sol_list

    def solve_Verlet(self, t_i, t_f, h=0.1):
        T = t_f - t_i
        n = len(self.odes_list)
        N = math.floor(T / h)

        t_list = np.zeros(N)
        sol_list = np.zeros((N, n))

        #First terms
        t = t_i
        sol_list[0] = self.init_vals_list
        t_list[0] = t

        #Second terms
        for i in range(math.floor(n / 2)):
            F_i = self.odes_list[i + 1](t, sol_list[0])
            
            sol_list[1, 2*i] = sol_list[0, 2*i] + h*sol_list[0, 2*i + 1] + (math.pow(h, 2) / 2)*F_i
            sol_list[1, 2*i + 1] = sol_list[0, 2*i + 1] + h*F_i

        t += h
        t_list[1] = t

        for i in range(1, N - 1):
            for j in range(math.floor(n / 2)):
                F_ij = self.odes_list[j + 1](t, sol_list[i])

                sol_list[i + 1, 2*j] = 2*sol_list[i, 2*j] - sol_list[i - 1, 2*j] + math.pow(h, 2)*F_ij
                sol_list[i + 1, 2*j + 1] = sol_list[i, 2*j + 1] + h*F_ij

            t += h
            t_list[i + 1] = t

        return t_list, sol_list

def plot(solutions):

    def plot_in_time(ax):
        for label, (t_vals, x_Dx_vals) in solutions.items():
            x_vals = x_Dx_vals[0:, 0]
            ax.plot(t_vals, x_vals, label=label)
            
        ax.set_xlabel("t(s)")
        ax.set_ylabel("x(m)")
        ax.legend(loc="best")

    plt.close("sum")
    fig, ax = plt.subplots(num="sum")
    plot_in_time(ax)