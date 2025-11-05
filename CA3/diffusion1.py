#!/ usr / bin /env
'''
diffusion.py

Random-walk model of diffusion in a 2D environment.
Starts with Np= 1000 particles in a square grid centered at (100,100).

At each step, the program picks each particle and moves it (or not) one integer step in the x and y directions.  If the move would take the particle beyond the boundary space (200x200), them the particle bounces off the wall and moves the other direction.

The program plots the position of all particles after each step.

The program needs with one argument as input from the user: the number of steps to take, Nsteps. The code could be changed to have the number being assigned directly in the code (uncomment "Nsteps = 90" line and change that number if needed). 

'''

import sys
from pylab import *

import numpy as np
import matplotlib.pyplot as plt
import random
import csv


# Allow animation
ion()

# set up graph window
figure(figsize =(6,6))

# Define the 2D cooordinates of all atoms to be initially at point (100,100).
Np=1000
atoms = np.ones([Np,2])*100


# plot the initial configuration

line,= plt.plot(atoms[:,0],atoms[:,1],'.')
plt.xlim(0,200)
plt.ylim(0,200)
draw()

# Number of steps (iterations) to be read from screen
Nsteps = int(input("Enter number of steps: "))
print("Number of steps: ", Nsteps)

# Alternatively, specify Nsteps directly in the code (uncomment line below):
# Nsteps = 90



for i in range(Nsteps):
    # Go through all atoms
    for j in range(Np):
        
        # Move each atom (or not) in the x and/or y direction.
        atoms[j,0]+= random.randint(-1,1)
        atoms[j,1]+= random.randint(-1,1)
        
        # Check for boundary collision
        x,y = (atoms[j,0], atoms[j,1])
        
        if x == 200:
            atoms[j,0] = 198
        elif x == 0:
            atoms[j,0] = 2
            if y == 200:
                atoms[j,1] = 198
            elif y == 0:
                atoms[j,1] = 2

        # Plot atoms at current step
        line.set_xdata(atoms[:,0])
        line.set_ydata(atoms[:,1])        
        draw( )
        
    wait = input('Press return (on terminal) to advance to next iteration')
    

data = np.zeros((Np,2))    
for j in range(Np):
    data[j] = [atoms[j,0],atoms[j,1]]


file = open("atoms.csv", "w")
writer = csv.writer(file)

for row in data:
    #  print(row)
    writer.writerow(row)
file.close()



