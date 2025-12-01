import numpy as np
import matplotlib.pyplot as plt

#Load data
TrainMat = np.load('TrainDigits.npy')
TrainLab = np.load('TrainLabels.npy')
TestLab = np.load('TestLabels.npy')
TestMat = np.load('TestDigits.npy')

#####Task 2 #####
#Plot singular values and images for digits 3 and 8 using 400 training images

n = 400  #Number of training images used

#Compute SVD for each digit 0–9
digit_data = {}
for i in range(10):
    index = (TrainLab == i)
    Ai_all = TrainMat[:, index[0]]        #All images of digit i
    Ai = Ai_all[:, 0:n]                   #First n images

    #SVD
    Ui, Si, VTi = np.linalg.svd(Ai)

    digit_data[i] = {
        'A': Ai,
        'U': Ui,
        'S': Si,
        'VT': VTi
    }

#Plot singular values of digit d
def singular_values(d):
    plt.figure(figsize=(10, 4))
    plt.plot(digit_data[d]['S'], marker='x')
    plt.title(f"Singular Values - Digit {d}")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.grid(True)
    plt.show()

#Plot first 3 singular image vectors (U1, U2, U3)
def singular_images(d):
    plt.figure()
    plt.suptitle(f"Singular Images U1, U2 & U3 for Digit {d}")
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"U{i + 1} - Digit {d}")
        u_i = digit_data[d]['U'][:, i]
        u = np.reshape(u_i, (28, 28)).T
        plt.imshow(u, cmap='gray')
    plt.tight_layout()
    plt.show()

#Run plots for digits 3 and 8
singular_values(3)
singular_images(3)
singular_values(8)
singular_images(8)

#####Task 3 #####

#Store first 15 singular images (U columns) for each digit
U15_dict = {}
for i in range(10):
    U15_dict[i] = {
        'U15': digit_data[i]['U'][:, :15]
    }

#Compute accuracies for k = 5, ..., 15
def accuracies_with_k(k_start, k_stop):

    all_accuracies = []
    U15T_TestMat_dict = {}

    #Precompute U15.T @ TestMat for each digit
    for i in range(10):
        U15T_TestMat_dict[i] = U15_dict[i]['U15'].T @ TestMat  #(15, 40000)

    for k in range(k_start, k_stop):

        residuals_dict = {}

        #Compute residuals for each digit model
        for i in range(10):
            Uk = U15_dict[i]['U15'][:, :k]                #(784, k)
            UkT_TestMat = U15T_TestMat_dict[i][:k, :]    #(k, 40000)

            residuals = np.linalg.norm(TestMat - Uk @ UkT_TestMat, axis=0)
            residuals_dict[i] = residuals

        #Stack residual vectors rowwise
        R = np.vstack([residuals_dict[i] for i in range(10)])  #(10, 40000)

        #Prediction = class with smallest residual
        predictions = np.argmin(R, axis=0)

        #Compute accuracy per digit
        accuracies = np.array([
            np.mean(predictions[TestLab[0] == d] == d) * 100
            for d in range(10)
        ])

        all_accuracies.append(accuracies)

    return np.array(all_accuracies)


#Plot per-digit accuracy for different k
def plot_accuracies(k_start, k_stop):
    all_accuracies = accuracies_with_k(k_start, k_stop)
    for d in range(10):
        plt.plot(range(k_start, k_stop), all_accuracies[:, d], label=f"Digit {d}")

    plt.xlabel("Number of singular images k")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy of predictions per digit for different values of k")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

plot_accuracies(5, 16)
