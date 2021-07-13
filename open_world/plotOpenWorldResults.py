import matplotlib.pyplot as plt
import numpy as np


def harmonic_mean(a, b):
    return (2 * a * b) / (a + b)


closed_world_with_rejection_icarl = {'0.3': [0.699, 0.464, 0.312, 0.269, 0.2358],
                                     '0.4': [0.69, 0.4165, 0.25533333333333336, 0.21875, 0.1856],
                                     '0.5': [0.657, 0.352, 0.20533333333333334, 0.17825, 0.1424],
                                     '0.6': [0.627, 0.303, 0.16866666666666666, 0.142, 0.1044],
                                     '0.7': [0.583, 0.258, 0.133, 0.106, 0.076],
                                     '0.8': [0.54, 0.214, 0.09966666666666667, 0.07775, 0.0466]}
open_world_icarl = {'0.3': [0.0274, 0.2088, 0.4252, 0.5298, 0.6142], '0.4': [0.115, 0.41, 0.625, 0.7244, 0.7672],
                    '0.5': [0.2328, 0.5714, 0.7566, 0.8296, 0.8612], '0.6': [0.3526, 0.6898, 0.8432, 0.898, 0.9162],
                    '0.7': [0.4646, 0.7806, 0.8972, 0.9362, 0.9462], '0.8': [0.5756, 0.8542, 0.9358, 0.967, 0.972]}

closed_world_with_rejection_bic = {'0.3': [0.82, 0.73, 0.6293333333333333, 0.582, 0.5216],
                                   '0.4': [0.814, 0.7225, 0.615, 0.56425, 0.504],
                                   '0.5': [0.803, 0.709, 0.5946666666666667, 0.53825, 0.4712],
                                   '0.6': [0.784, 0.6845, 0.5673333333333334, 0.50075, 0.4348],
                                   '0.7': [0.76, 0.6545, 0.529, 0.46375, 0.388],
                                   '0.8': [0.724, 0.6155, 0.48233333333333334, 0.41775, 0.3386]}
open_world_bic = {'0.3': [0.0112, 0.0204, 0.0514, 0.0688, 0.1078], '0.4': [0.0448, 0.08, 0.1578, 0.1978, 0.2532],
                  '0.5': [0.1098, 0.18, 0.2888, 0.3348, 0.4112], '0.6': [0.217, 0.2892, 0.4174, 0.467, 0.5548],
                  '0.7': [0.3152, 0.4024, 0.5392, 0.5808, 0.6832], '0.8': [0.4172, 0.5178, 0.6546, 0.687, 0.7944]}

# Questo script è pensato per essere runnato su due dizionari: icarl_accuracies e bic_accuracies che memorizzano le testing accuracy nei tre casi (closed_with, closed without, openset) per ogni soglia
# open_world_harmonic_mean_icarl[str(0.5)] = [harmonic_mean(a,b) for a,b in zip(closed_word_with_rejection_accuracy[str(threeshold)], open_set_accuracy[str(threeshold)])]
# open_world_harmonic_mean_bic[str(0.5)] = [harmonic_mean(a,b) for a,b in zip(closed_word_with_rejection_accuracy[str(threeshold)], open_set_accuracy[str(threeshold)])]

# --- Naive Strategy : treeshold 0.5
# plot iCaRL vs BiC in termini di open world armonic mean

x = np.arange(10, 60, 10)  # genero vettore 10,20,30,40,50

open_world_harmonic_mean_icarl = {}
open_world_harmonic_mean_bic = {}
rejection_global_treesholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for threeshold in rejection_global_treesholds:
    open_world_harmonic_mean_icarl[str(threeshold)] = [harmonic_mean(a, b) for a, b in
                                                       zip(closed_world_with_rejection_icarl[str(threeshold)],
                                                           open_world_icarl[str(threeshold)])]
    open_world_harmonic_mean_bic[str(threeshold)] = [harmonic_mean(a, b) for a, b in
                                                     zip(closed_world_with_rejection_bic[str(threeshold)],
                                                         open_world_bic[str(threeshold)])]

# plt.show()

# --- Treshold comparison
# Due plot (uno iCaRL uno BiC) in cui si comparano le harmonic mean delle rispettive treeshold
# quindi per ogni treshold prima della funzione di plot bisogna mediare i risultati ottenuti con i diversi seed
# nella funzione per ciascun metodo scorriamo le harmonic mean per pgni treshold e le plottiamo

print("Global treesholds: ", rejection_global_treesholds)
print("\n")
print("ICARL) CW: ", closed_world_with_rejection_icarl, "\n OS: ", open_world_icarl)
print("\n")
print("BIC) CW: ", closed_world_with_rejection_bic, "\n OS: ", open_world_bic)
print("\n")
# fig, ax = plt.subplots(1,6)
for i, threeshold in enumerate(rejection_global_treesholds):
    print(f"Harmonic means ICARL vs BIC with {threeshold}")
    print("\n")
    print("Harmonic mean ICARL: ", open_world_harmonic_mean_icarl[str(threeshold)])
    print("Harmonic mean BIC: ", open_world_harmonic_mean_bic[str(threeshold)])
    print("\n")
    open_world_harmonic_mean_icarl[str(threeshold)] = [harmonic_mean(a, b) for a, b in
                                                       zip(closed_world_with_rejection_icarl[str(threeshold)],
                                                           open_world_icarl[str(threeshold)])]
    open_world_harmonic_mean_bic[str(threeshold)] = [harmonic_mean(a, b) for a, b in
                                                     zip(closed_world_with_rejection_bic[str(threeshold)],
                                                         open_world_bic[str(threeshold)])]
    plt.plot(x, open_world_harmonic_mean_icarl[str(threeshold)], '-o')
    plt.plot(x, open_world_harmonic_mean_bic[str(threeshold)], '-o')
    plt.legend()
    # ax[i].set_title(str(threeshold))
    plt.show()
