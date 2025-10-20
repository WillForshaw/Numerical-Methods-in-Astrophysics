import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

def getAcc( pos, mass, G, softening ):
	"""
    Calculate the acceleration on each particle due to Newton's Law 
	pos  is an N x 3 matrix of positions
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	softening is the softening length
	a is N x 3 matrix of accelerations
	"""
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
	inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

	ax = G * (dx * inv_r3) @ mass
	ay = G * (dy * inv_r3) @ mass
	az = G * (dz * inv_r3) @ mass
	
	a = np.hstack((ax,ay,az))

	return a
	

def getEnergy( pos, vel, mass, G ):
	"""
	Get kinetic energy (KE) and potential energy (PE) of simulation
	"""
	KE = 0.5 * np.sum(np.sum( mass * vel**2 ))

	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
	inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

	PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
	
	return KE, PE;


def main():
	""" N-body simulation for two star clusters """
	
	
	t         = 0      # current time of the simulation
	tEnd      = 10.0   # time at which simulation ends
	dt        = 0.01   # timestep
	softening = 0.05   # softening length
	G         = 1.0    # Newton's Gravitational Constant
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	data = pd.read_csv('cluster_ICs.csv')
	pos_cluster = data[['x','y','z']].values
	vel_cluster = data[['vx','vy','vz']].values
	N1 = len(pos_cluster)
	N2 = N1
	N = N1 + N2
	
	M_total = 1.024
	mass = (M_total / N) * np.ones((N,1))
	

	offset = 2.0           # separation between clusters
	v_rel  = 0.3          # relative velocity

	pos1 = pos_cluster + np.array([-offset/2, 0, 0])
	pos2 = pos_cluster + np.array([ offset/2, 0, 0])
	vel1 = vel_cluster + np.array([ v_rel, 0, 0])
	vel2 = vel_cluster + np.array([-v_rel, 0, 0])

	pos = np.vstack((pos1, pos2))
	vel = np.vstack((vel1, vel2))

	
	# Calculate initial acceleration and energy
	
	acc = getAcc(pos, mass, G, softening)
	KE, PE = getEnergy(pos, vel, mass, G)

	Nt = int(np.ceil(tEnd/dt))
	pos_save = np.zeros((N,3,Nt+1))
	pos_save[:,:,0] = pos
	KE_save = np.zeros(Nt+1)
	PE_save = np.zeros(Nt+1)
	KE_save[0], PE_save[0] = KE, PE
	t_all = np.arange(Nt+1)*dt
	
	
	fig = plt.figure(figsize=(4,5), dpi=80)
	grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
	ax1 = plt.subplot(grid[0:2,0])
	ax2 = plt.subplot(grid[2,0])
	
	
	for i in range(Nt):
		# ½ kick
		vel += acc * dt/2
		# drift
		pos += vel * dt
		# update accelerations
		acc = getAcc(pos, mass, G, softening)
		# ½ kick
		vel += acc * dt/2
		# update time
		t += dt
		# energy
		KE, PE = getEnergy(pos, vel, mass, G)
		pos_save[:,:,i+1] = pos
		KE_save[i+1], PE_save[i+1] = KE, PE

		if plotRealTime or (i == Nt-1):
			plt.sca(ax1)
			plt.cla()
			xx = pos_save[:,0,max(i-50,0):i+1]
			yy = pos_save[:,1,max(i-50,0):i+1]
			plt.scatter(xx,yy,s=1,color=[.7,.7,1])
			plt.scatter(pos[:,0],pos[:,1],s=10,color='green')
			ax1.set(xlim=(-4,4), ylim=(-3,3))
			ax1.set_aspect('equal','box')
			#ax1.set_xticks([-6,-3,0,3,6])
			#ax1.set_yticks([-6,-3,0,3,6])
			
			plt.sca(ax2)
			plt.cla()
			plt.scatter(t_all,KE_save,color='red',s=1,label='KE' if i == Nt-1 else "")
			plt.scatter(t_all,PE_save,color='blue',s=1,label='PE' if i == Nt-1 else "")
			plt.scatter(t_all,KE_save+PE_save,color='black',s=1,label='Etot' if i == Nt-1 else "")
			ax2.set_xlim(0,tEnd)
			ax2.set_ylim(-0.5,0.5)
			ax2.relim()
			ax2.autoscale_view()
			#ax2.set_aspect(0.007)
			plt.pause(0.001)
			
	
	plt.sca(ax2)
	plt.xlabel('time')
	plt.ylabel('energy')
	ax2.legend(loc='upper right')
	plt.savefig('nbody_cluster_collision.png',dpi=240)
	plt.show()
        
	return 0
  

if __name__== "__main__":
	main()


from scipy.optimize import curve_fit

# Load initial cluster data
data = pd.read_csv('cluster_ICs.csv')
pos = data[['x', 'y', 'z']].values
r = np.linalg.norm(pos, axis=1)


bins = np.linspace(0, np.max(r), 40)
counts, edges = np.histogram(r, bins=bins)
shell_vol = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)
n_r = counts / shell_vol
r_mid = 0.5 * (edges[1:] + edges[:-1])

# Remove zero-count bins (avoids log(0) and divide-by-zero errors)
mask = n_r > 0
r_mid = r_mid[mask]
n_r = n_r[mask]

def king_profile(r, n0, Rc):
    return n0 * (1 + (r / Rc)**2)**(-1.5)


popt, pcov = curve_fit(king_profile, r_mid, n_r, p0=[n_r.max(), 0.2])
n0_fit, Rc_fit = popt
n0_err, Rc_err = np.sqrt(np.diag(pcov))

print(f"Fitted King parameters:")
print(f"  n0 = ({n0_fit:.3e} ± {n0_err:.3e})")
print(f"  Rc = ({Rc_fit:.3f} ± {Rc_err:.3f})")


plt.figure(figsize=(6,4))
plt.scatter(r_mid, n_r, s=20, color='navy', label='Cluster data')
plt.plot(r_mid, king_profile(r_mid, *popt), 'r-', lw=2,
         label=fr'King fit: $n_0={n0_fit:.2e}\pm{n0_err:.2e}$, $R_c={Rc_fit:.3f}\pm{Rc_err:.3f}$')
plt.yscale('log')
plt.xlabel('Radius $r$')
plt.ylabel('Number density $n(r)$')
#plt.title('Star Cluster Density Profile with King Model Fit')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
