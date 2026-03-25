import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========================
# 1. Parameter Configuration
# ========================
np.random.seed(42)
noise_levels = np.linspace(0.0, 0.1, 100)
criticalities = np.linspace(0.0, 1.0, 100)

# ========================
# 2. Physical Model (Error Rate + Resource Consumption)
# ========================
def error_model(protocol: str, gamma: float) -> float:
    if protocol == 'Steane':
        return 0.0002 * gamma + 0.0005 * np.exp(9.0 * gamma)
    elif protocol == 'Bacon-Shor':
        return 0.0001 * gamma + 0.0006 * np.exp(4.0 * gamma)
    else:
        raise ValueError("Invalid protocol")

def resource_model(protocol: str, gamma: float) -> float:
    """Computational resource consumption model (qubit-hours)"""
    if protocol == 'Steane':
        return 10 + 50 * gamma  # Higher base cost but scales better
    elif protocol == 'Bacon-Shor':
        return 20 + 30 * gamma  # Lower base cost but scales worse
    else:
        raise ValueError("Invalid protocol")

def qudit_resources(qubit_resources: np.ndarray) -> np.ndarray:
    return qubit_resources / 2

def accuracy_from_error_rate(error_rate):
    return 1 - error_rate

# ========================
# 3. Dynamic Strategy
# ========================
def dynamic_selector(gamma: float, crit: float) -> tuple:
    if gamma < 0.04:
        return (error_model('Steane', gamma), resource_model('Steane', gamma))
    elif gamma >= 0.07:
        return (error_model('Bacon-Shor', gamma), resource_model('Bacon-Shor', gamma))
    else:
        transition = 1 / (1 + np.exp(-10*(crit - 0.5)))
        e_steane = error_model('Steane', gamma)
        e_bs = error_model('Bacon-Shor', gamma)
        r_steane = resource_model('Steane', gamma)
        r_bs = resource_model('Bacon-Shor', gamma)
        return (
            transition * e_bs + (1 - transition) * e_steane,
            transition * r_bs + (1 - transition) * r_steane
        )

# ========================
# 4. Generate Grid Data
# ========================
X, Y = np.meshgrid(noise_levels, criticalities)

# Error rates and resources
Z_static_steane = np.array([[error_model('Steane', x) for x in noise_levels] for _ in criticalities])
Z_static_bs = np.array([[error_model('Bacon-Shor', x) for x in noise_levels] for _ in criticalities])

dynamic_results = np.array([[dynamic_selector(x, y) for x in noise_levels] for y in criticalities])
Z_dynamic = dynamic_results[:, :, 0]
Z_dynamic_res = dynamic_results[:, :, 1]

R_static_steane = qudit_resources(np.array([[resource_model('Steane', x) for x in noise_levels] for _ in criticalities]))
R_static_bs = qudit_resources(np.array([[resource_model('Bacon-Shor', x) for x in noise_levels] for _ in criticalities]))
Z_dynamic_res_qudit = qudit_resources(Z_dynamic_res)

# Calculate accuracies
Z_static_steane_accuracy = np.vectorize(accuracy_from_error_rate)(Z_static_steane)
Z_static_bs_accuracy = np.vectorize(accuracy_from_error_rate)(Z_static_bs)
Z_dynamic_accuracy = np.vectorize(accuracy_from_error_rate)(Z_dynamic)

# ========================
# 5. Visualization
# ========================
plt.style.use('seaborn-v0_8')

fig = plt.figure(figsize=(24, 12))

# Use gradient colors and remove color bars
def plot_surface_with_gradient(ax, X, Y, Z, title, zlabel, zlim=None):
    surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                           rstride=2, cstride=2,
                           linewidth=0.1, antialiased=True)
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel('Noise (γ)', fontsize=8, labelpad=5)
    ax.set_ylabel('Criticality', fontsize=8, labelpad=5)
    ax.set_zlabel(zlabel, fontsize=8, labelpad=5)
    ax.view_init(25, -45)
    if zlim is not None:
        ax.set_zlim(*zlim)

# Error Rate Visualization
for i, (data, title) in enumerate(zip(
    [Z_static_steane, Z_static_bs, Z_dynamic],
    ['Static Steane Code', 'Static Bacon-Shor Code', 'Dynamic QECC Strategy']
)):
    ax = fig.add_subplot(231+i, projection='3d')
    plot_surface_with_gradient(ax, X, Y, data,
                               f"{title}\nError Rate: {data.min():.5f}-{data.max():.5f}",
                               'Error Rate', (0, 0.002))

# Accuracy Visualization
for i, (data, title) in enumerate(zip(
    [Z_static_steane_accuracy, Z_static_bs_accuracy, Z_dynamic_accuracy],
    ['Static Steane Code Accuracy', 'Static Bacon-Shor Code Accuracy', 'Dynamic QECC Strategy Accuracy']
)):
    ax = fig.add_subplot(234+i, projection='3d')
    plot_surface_with_gradient(ax, X, Y, data,
                               f"{title}\nAccuracy: {data.min():.5f}-{data.max():.5f}",
                               'Accuracy', (0.998, 1))

# Improve layout and remove unnecessary elements
plt.tight_layout()
plt.savefig('dynamic_strategy_with_accuracies_optimized.pdf', dpi=300, bbox_inches='tight')
plt.show()