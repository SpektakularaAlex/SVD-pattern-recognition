import numpy as np
import matplotlib.pyplot as plt


TrainMat = np.load("TrainDigits.npy")
TrainLab = np.load("TrainLabels.npy")


d = TrainMat[:,6] # The first digit in the training set
D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
#plt.imshow(D, cmap ="gray") # Plot of the digit

length = 400
digit = 3

index = (TrainLab == digit); # find train digits of type 3
A_i = TrainMat[:,index[0]] # all train digits of type 3
A_i = A_i[:,0:length] # the first 400 train digits of type 

U_i, S_i, Vt_i = np.linalg.svd(A_i)



"""
for i in range(3):
        u_img = np.reshape(U_i[:, i], (28,28)).T
        plt.figure()
        plt.imshow(u_img, cmap='gray')
        plt.title(f'Digit {digit} - u{i+1}')
        plt.axis('off')
        plt.show()
"""
    
for i in range(3):   
    
    ui_vector = U_i[:,i]
    ui_vector = np.reshape(ui_vector, (28, 28)).T
    plt.imshow(ui_vector, cmap ="gray")
    plt.figure()



"""
fig, axs = plt.subplots(1, 3, figsize=(8, 6), sharex=True)

# Top plot: Deterministic
axs[0].plot(sol.t, sol.y[7], color='black', label='Deterministic R')
axs[0].set_ylabel('Molecules [mol]', fontsize=12)
axs[0].set_title('Deterministic solution (delta_r = 0.05)', fontsize=25)
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Bottom plot: Stochastic
axs[1].plot(t, R, color='gray', label='Stochastic R')  
axs[1].set_xlabel('Time [h]')
axs[1].set_ylabel('Molecules [mol]', fontsize=12)
axs[1].set_title('Stochastic solution (delta_r = 0.05)', fontsize=25)
axs[1].legend(loc='upper right')
axs[1].grid(True)

plt.tight_layout()
plt.show()
"""

"""
k = 10

U_k = U_i[:, :k]

U_k = np.reshape(U_k, (28, 28)).T

plt.imshow(U_k, cmap ="gray")
    
"""



"""
sigma_array = np.zeros(length)
for i in range(len(S_3)):
    if S_3[i] != 0:
        sigma_array[i] = S_3[i]
        


"<--- Här har vi sigmavärdena --->"
x = np.arange(0, length)

plt.figure(figsize=(10, 5))
plt.plot(x, sigma_array, marker = 'x', color='blue', label = 'sigma')

plt.xlabel('index'); plt.ylabel('sigma')
plt.title('sigma from diagonalmatrix')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()
"""
