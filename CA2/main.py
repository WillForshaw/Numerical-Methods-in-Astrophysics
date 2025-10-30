#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# parameters
o = 10
p = 28
B = 8 / 3

# Time parameters
dt = 0.01          # time step
N = 10000          # number of iterations
T = N * dt         # total integration time
t_eval = np.linspace(0, T, N)  # time points where we want the solution


def Lorenz(t, coord):   # function that takes in time and an array of [x,y,z] coords at a given time
    x, y, z = coord
    dxdt = o * (y - x)
    dydt = x * (p - z) - y
    dzdt = x * y - B * z
    return [dxdt, dydt, dzdt]


# Initial conditions [x,y,z] at time = 0
init_coords = [1, 1, 1]

# Integrate from t = 0 to T
sol = solve_ivp(Lorenz, [0, T], init_coords, t_eval=t_eval)

# Extract results
x, y, z = sol.y


#midpoint restart
mid_index = N // 2
mid_time = t_eval[mid_index] # find midpoit in time
mid_state = [x[mid_index], y[mid_index], z[mid_index]]   # find midpoint for x, y, z 

t_eval_restart = np.linspace(mid_time, T, N - mid_index)  # midpoimt in time -> T in N/2 steps 
sol_restart = solve_ivp(Lorenz, [mid_time, T], mid_state, t_eval=t_eval_restart, method='RK45')
x_r, y_r, z_r = sol_restart.y

# x(t)
plt.figure(figsize=(7, 4))
plt.plot(sol.t[mid_index:], x[mid_index:], 'r', label='Full Run (2nd Half)')
plt.plot(sol_restart.t, x_r, 'k--', label='Restarted Run')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('x(t)', fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim(T/2-10*dt, T/2 + 1500*dt)
#plt.title('Lorenz System: x(t) — Full vs Restarted')
plt.legend()
plt.tight_layout()
plt.savefig("Restartx.png")
plt.show()

# y(t)
plt.figure(figsize=(7, 4))
plt.plot(sol.t[mid_index:], y[mid_index:], 'g', label='Full Run (2nd Half)')
plt.plot(sol_restart.t, y_r, 'k--', label='Restarted Run')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('y(t)', fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim(T/2-10*dt, T/2 + 1500*dt)
#plt.title('Lorenz System: y(t) — Full vs Restarted')
plt.legend()
plt.tight_layout()
plt.savefig("Restarty.png")
plt.show()

# z(t)
plt.figure(figsize=(7, 4))
plt.plot(sol.t[mid_index:], z[mid_index:], 'b', label='Full Run (2nd Half)')
plt.plot(sol_restart.t, z_r, 'k--', label='Restarted Run')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('z(t)', fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim(T/2-10*dt, T/2 + 1500*dt)
#plt.title('Lorenz System: z(t) — Full vs Restarted')
plt.legend()
plt.tight_layout()
plt.savefig("Restartz.png")
plt.show()

#%%
def Lorenz_receiver(t, coords):
    # coords = [x, y, z, y_r, z_r]
    x, y, z, y_r, z_r = coords
    o = 10
    p = 28
    B = 8 / 3

    # Driver system
    dxdt = o * (y - x)
    dydt = x * (p - z) - y
    dzdt = x * y - B * z

    # Receiver system (gets x from driver)
    dydt_r = x * (p - z_r) - y_r
    dzdt_r = x * y_r - B * z_r

    return [dxdt, dydt, dzdt, dydt_r, dzdt_r]


init_coords2 = [3,4,1,-10,-6]
sol_receiver =  solve_ivp(Lorenz_receiver, [0, T], init_coords2, t_eval=t_eval)
x, y, z, y_r, z_r = sol_receiver.y


plt.figure(figsize=(7, 4))
plt.plot(sol_receiver.t, y, label="Driver y'", color='red')
plt.plot(sol_receiver.t, y_r, label="Receiver y", color='black', linestyle='--')
plt.xlabel("Time", fontsize = 16)
plt.ylabel("y(t)", fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim(0, sol_receiver.t[500])
#plt.title("Driver vs Receiver (y-components)")
plt.legend()
plt.tight_layout()
plt.savefig("Drivery.png")
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(sol_receiver.t, z, label="Driver z'", color='blue')
plt.plot(sol_receiver.t, z_r, label="Receiver z", color='black', linestyle='--')
plt.xlabel("Time", fontsize = 16)
plt.ylabel("z(t)", fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim(0, sol_receiver.t[500])
#plt.title("Driver vs Receiver (z-components)")
plt.legend()
plt.tight_layout()
plt.savefig("Driverz.png")
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(sol_receiver.t, x, label="Driver x'", color='blue')
plt.plot(sol_receiver.t, x, label="Receiver x", color='black', linestyle='--')
plt.xlabel("Time", fontsize = 16)
plt.ylabel("x(t)",fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim(0, sol_receiver.t[500])
#plt.title("Driver vs Receiver (x-components)")
plt.legend()
plt.tight_layout()
plt.savefig("Driverx.png")
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(y, z, color='r', alpha=0.6, label="Driver")
plt.plot(y_r, z_r, color='k', alpha=0.6, linestyle='--', label="Receiver")
plt.xlabel("y", fontsize = 16)
plt.ylabel("z", fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
#plt.title("Synchronization y-z plane")
plt.legend()
plt.tight_layout()
plt.savefig("Drivery_z.png")
plt.show()
# %%
