import numpy as np
import json
import matplotlib.pyplot as plt

def create_vqe_plots(potential, base_path, folder, cut_off_list):
    # Load all data and create graphs
    print("Creating plots")
    data_dict = {}

    for n in cut_off_list:
        file_path = base_path.format(potential, folder) + potential + "_" + str(n) + ".json"
        with open(file_path, 'r') as json_file:
            data_dict[f'c{n}'] = json.load(json_file)


    #Create and save plots
    num_cutoffs = len(cut_off_list)
    nrows = int(np.ceil(num_cutoffs/2))
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(30, 5*nrows))
    axes = axes.flatten()

    for idx, (cutoff, cutoff_data) in enumerate(data_dict.items()):
        
        results = cutoff_data['results']
        x_values = range(len(results))

        # Calculating statistics
        mean_value = np.mean(results)
        median_value = np.median(results)
        min_value = np.min(results)

        # Creating the plot
        ax = axes[idx]
        ax.plot(x_values, results, marker='o', label='Energy Results')

        # Plot mean, median, and min lines
        ax.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean = {mean_value:.6f}')
        ax.axhline(y=median_value, color='g', linestyle='-', label=f'Median = {median_value:.6f}')
        ax.axhline(y=min_value, color='b', linestyle='-.', label=f'Min = {min_value:.6f}')

        ax.set_ylim(min_value - 0.01, max(results) + 0.01)
        ax.set_xlabel('Run')
        ax.set_ylabel('Ground State Energy')
        ax.set_title(f"{potential}: Cutoff = {cutoff_data['cutoff']}")
        ax.legend()
        ax.grid(True)

    # Hide any remaining unused axes
    for idx in range(num_cutoffs, len(axes)):
        fig.delaxes(axes[idx])

    print("Saving plots")
    plt.tight_layout()
    plt.savefig(base_path.format(potential, folder) + "results.png")

    print("Done")


def create_vqd_plots(data, path, show=False):

    num_VQD = data['num_VQD']
    converged_energies = data['converged_energies']

    transposed_energies = list(zip(*converged_energies))
    medians = [np.median(energies) for energies in transposed_energies]

    # Plotting
    plt.figure(figsize=(8, 6))

    for i, energies in enumerate(transposed_energies):
        line, = plt.plot(range(1, num_VQD + 1), energies, marker='o', linestyle='-', label=f"$E_{{{i}}} \\approx {medians[i]:.3f}$")
        plt.axhline(data['exact_eigenvalues'][i], color = line.get_color(), linestyle='--', linewidth=0.5, label=f"$E_{{{i}}}^{{\\text{{ex}}}} = {data['exact_eigenvalues'][i]:.3f}$")

    plt.xlabel("VQD Run")
    plt.ylabel("Energy")
    plt.title(f"{data['potential']}: Cutoff = {data['cutoff']}")
    plt.xticks(range(1, num_VQD + 1))
    plt.legend()
    plt.tight_layout()

    if show == True:
        plt.show()
    else:
        print("Saving plots")
        plt.savefig(path + "results.png")

    print("Done")