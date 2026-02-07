def plot_matplotlib_molecule(topology, positions, figsize=(4, 4), atom_scale=50):
    """
    Plots an OpenMM molecule using Matplotlib 3D. While not being visually the best, it provides a simple and reliable
    way to inspect molecular structures (e.g., for debugging).

    Parameters
    ----------
    topology : openmm.app.Topology
        OpenMM topology object
    positions : openmm.unit.Quantity or numpy.ndarray
        Atomic positions in nanometers
    figsize : tuple
        Figure size
    atom_scale : float
        Scale factor for atom size
    """
    import simtk.unit as unit

    import matplotlib.pyplot as plt
    import numpy as np

    # Define a simple color map for atom elements
    ATOM_COLORS = {
        "H": "white",
        "C": "grey",
        "N": "blue",
        "O": "red",
        "S": "yellow",
        "P": "orange",
        "F": "green",
        "CL": "green",
        "BR": "brown",
        "I": "purple",
        # fallback
        "DEFAULT": "pink",
    }

    # Convert positions to numpy array (in nm)
    if hasattr(positions, "value_in_unit"):
        pos = np.array(positions.value_in_unit(unit.nanometers))
        print("convert to nanometers")
    else:
        pos = np.array(positions)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot atoms
    for atom, coord in zip(topology.atoms(), pos):
        element = atom.element.symbol.upper() if atom.element else "DEFAULT"
        color = ATOM_COLORS.get(element, ATOM_COLORS["DEFAULT"])
        ax.scatter(
            coord[0],
            coord[1],
            coord[2],
            s=atom_scale,
            color=color,
            edgecolors="k",
            depthshade=True,
        )

    # Plot bonds
    for bond in topology.bonds():
        i = bond[0].index
        j = bond[1].index
        xyz = np.array([pos[i], pos[j]])
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="black", linewidth=1)

    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")

    plt.show()
