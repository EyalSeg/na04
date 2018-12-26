import numpy as np
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def get_lud(matrix):
    lu = np.copy(matrix)
    d = np.zeros(matrix.shape)
    dInverse = np.zeros(matrix.shape)

    for rowIndex, row in enumerate(matrix):
        lu[rowIndex][rowIndex] = 0
        d[rowIndex][rowIndex] = row[rowIndex]
        dInverse[rowIndex][rowIndex] = 1/row[rowIndex]

    return lu, d, dInverse


def jacobi_step(a_matrix, dInv, b_vector, previous_step):
    ax_prev = np.dot(a_matrix, previous_step)
    b_minus_ax = np.subtract(b_vector, ax_prev)

    return np.add(previous_step, np.dot(dInv, b_minus_ax))

def l2_norm(vector):
    return math.sqrt(sum([math.pow(item, 2) for item in vector]))

def get_residue_norm(a_matrix, b_vector, step_result):
    ax = np.dot(a_matrix, step_result)
    ax_minus_b = np.subtract(ax, b_vector)

    return l2_norm(ax_minus_b)



# Solves Ax = b
def jacobi(a_matrix, b_vector, initial_guess, max_residue):
    lu, d, dInv = get_lud(a_matrix)

    step_results = [initial_guess]
    residues = [get_residue_norm(a_matrix, b_vector, initial_guess)]

    while residues[-1] > max_residue:
        nextIteration = jacobi_step(a_matrix, dInv, b_vector, step_results[-1])

        step_results.append(nextIteration)
        residues.append(get_residue_norm(a_matrix, b_vector, nextIteration))

    return step_results, residues

def get_error_norms(a_matrix, b_vector, results_steps):
    actual_results = np.linalg.solve(a_matrix, b_vector)
    error_values = [np.subtract(result_step, actual_results)
            for result_step in results_steps]

    return [l2_norm(error) for error in error_values]


def plot(step_results, residues, errors):
    f, axarr = plt.subplots(3, 1)

    axarr[0].set_title("Residue norms")
    axarr[0].semilogy(residues)

    axarr[1].set_title("Error norms")
    axarr[1].semilogy(errors)

    axarr[2].set_title("Results")
    axarr[2].plot([tuple[0] for tuple in step_results], color='red', label="x1")
    axarr[2].plot([tuple[1] for tuple in step_results], color='yellow', label="x2")
    axarr[2].plot([tuple[2] for tuple in step_results], color='green', label="x3")
    axarr[2].plot([tuple[3] for tuple in step_results], color='blue', label="x4")
    axarr[2].plot([tuple[4] for tuple in step_results], color='purple', label="x5")
    plt.show()

a = np.array([[-5, 0.2, 0.2, 0.2, 0.2],
              [0.2, -5, 0.2, 0.2, 0.2],
              [0.2, 0.2, -5, 0.2, 0.2],
              [0.2, 0.2, 0.2, -5, 0.2],
              [0.2, 0.2, 0.2, 0.2, -5]
              ])
b = np.array([1, 2, 3, 4, 5])
initial_guess = np.array([1, 1, 1, 1, 1])
lu, d, dInv = get_lud(a)

results, residues = jacobi(a, b, initial_guess, 0.0000001)
errors = get_error_norms(a, b, results)
print(results[-1])
print(len(results))
plot(results, residues, errors)


