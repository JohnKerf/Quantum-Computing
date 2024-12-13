import numpy as np
import json
import matplotlib.pyplot as plt

def create_vqe_plots(potential, base_path, cut_off_list, individual, converged_only=False):

    # Load all data from folder
    print("Creating plots")
    data_dict = {}

    for n in cut_off_list:
        file_path = base_path.format(potential) + potential + "_" + str(n) + ".json"
        with open(file_path, 'r') as json_file:
            data_dict[f'c{n}'] = json.load(json_file)

    if not individual:
    # Create axes for plots
        num_cutoffs = len(cut_off_list)
        nrows = int(np.ceil(num_cutoffs/2))
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(30, 5*nrows))
        axes = axes.flatten()

    for idx, (_, data) in enumerate(data_dict.items()):

        # Remove data that didnt converge if required
        if converged_only:
            successful_indices = [i for i, success in enumerate(data['success']) if success]
            results = [data['results'][i] for i in successful_indices]
        else:
            results = data['results']
        
        cutoff = data['cutoff']
        exact = np.min(data['exact_eigenvalues'])
        x_values = range(len(results))

        # Calculating statistics
        mean_value = np.mean(results)
        median_value = np.median(results)
        min_value = np.min(results)

        if individual:

            # Save individual plot
            print(f"Saving individual plot for cutoff {cutoff}")

            plt.figure(figsize=(15,10))
            plt.plot(x_values, results, marker='o', label='Energy Results')

            # Plot mean, median, min and exact lines
            plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean = {mean_value:.6f}')
            plt.axhline(y=median_value, color='g', linestyle='-', label=f'Median = {median_value:.6f}')
            plt.axhline(y=min_value, color='b', linestyle='-.', label=f'Min = {min_value:.6f}')
            plt.axhline(y=exact, color='orange', linestyle=':', label=f'Exact = {exact:.6f}')

            plt.ylim(min_value - 0.01, max(results) + 0.01)
            plt.xlabel('Run')
            plt.ylabel('Ground State Energy')
            plt.title(f"{potential}: Cutoff = {cutoff}")
            plt.legend()
            plt.grid(True)
            plt.savefig(base_path.format(potential) + f"cutoff_{cutoff}_plot.png")
            plt.close()

        else:

            # Add each plot to same figure
            ax = axes[idx]
            ax.plot(x_values, results, marker='o', label='Energy Results')

            # Plot mean, median, min and exact lines
            ax.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean = {mean_value:.6f}')
            ax.axhline(y=median_value, color='g', linestyle='-', label=f'Median = {median_value:.6f}')
            ax.axhline(y=min_value, color='b', linestyle='-.', label=f'Min = {min_value:.6f}')
            ax.axhline(y=exact, color='orange', linestyle=':', label=f'Exact = {exact:.6f}')

            ax.set_ylim(min_value - 0.01, max(results) + 0.01)
            ax.set_xlabel('Run')
            ax.set_ylabel('Ground State Energy')
            ax.set_title(f"{potential}: Cutoff = {cutoff}")
            ax.legend()
            ax.grid(True)

    

    if not individual:

        # Hide any remaining unused axes
        for idx in range(num_cutoffs, len(axes)):
            fig.delaxes(axes[idx])

        print("Saving combined plots")
        plt.tight_layout()
        plt.savefig(base_path.format(potential) + "results.png")

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