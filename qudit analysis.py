import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_theme(style="whitegrid", font_scale=1.2)

# Parameter settings
m = np.arange(2, 21)
k = 3
d_code = 4
p = np.linspace(0.001, 0.1, 100)
code_distances = [2, 4, 6]

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# ----------------------------
# Subplot 1: Quantum Resource Requirements
# ----------------------------
qubits_qubit = np.ceil(m)
qubits_qudit = np.ceil(m / 2)
axs[0,0].plot(m, qubits_qudit, 'r-', linewidth=2, label='4D-Qudit')
axs[0,0].plot(m, qubits_qubit, 'b--', linewidth=2, label='2D-Qubit')
axs[0,0].set_xlabel('Data Dimension ($m$)', fontsize=12)
axs[0,0].set_ylabel('Qudits/Qubits Required', fontsize=12)
axs[0,0].set_title('(a) Quantum Units for $m$D Data', fontsize=14)
axs[0,0].grid(True, linestyle='--', alpha=0.6)
axs[0,0].legend()

# ----------------------------
# Subplot 2: QFT Complexity Comparison
# ----------------------------
log2_m = np.log2(m)
qubit_complexity = log2_m ** 2
qudit_complexity = np.log(m) / np.log(4)
axs[0,1].plot(m, qudit_complexity, 'r-', linewidth=2, label='4D-Qudit')
axs[0,1].plot(m, qubit_complexity, 'b--', linewidth=2, label='2D-Qubit')
axs[0,1].set_xlabel('Data Dimension ($m$)', fontsize=12)
axs[0,1].set_ylabel('QFT Gate Complexity', fontsize=12)
axs[0,1].set_title('(b) QFT Complexity Scaling', fontsize=14)
axs[0,1].grid(True, linestyle='--', alpha=0.6)
axs[0,1].legend()

# ----------------------------
# Subplot 3: Fault-Tolerance Redundancy
# ----------------------------
d = 4
m = np.arange(2, 21)
qubit_redundancy = d**2 * m
qudit_redundancy = (d * m) / 2

axs[0,2].plot(m, qudit_redundancy, 'r-', linewidth=2, label='4D-Qudit')
axs[0,2].plot(m, qubit_redundancy, 'b--', linewidth=2, label='2D-Qubit')
axs[0,2].set_xlabel('Data Dimension ($m$)', fontsize=12)
axs[0,2].set_ylabel('Physical Qubits Required', fontsize=12)
axs[0,2].set_title(f'(c) Redundancy for Fault Tolerance (d={d})', fontsize=14)
axs[0,2].grid(True, linestyle='--', alpha=0.6)
axs[0,2].legend()

# ----------------------------
# Subplot 4: Logical Error Rate Comparison
# ----------------------------
logical_error_qubit = p ** (d_code/2)
logical_error_qudit = (p/4) ** (d_code/2)
axs[1,0].semilogy(p, logical_error_qubit, 'b--', linewidth=2, label='2D-Qubit')
axs[1,0].semilogy(p, logical_error_qudit, 'r-', linewidth=2, label='4D-Qudit')
axs[1,0].set_xlabel('Physical Error Rate ($p$)', fontsize=12)
axs[1,0].set_ylabel('Logical Error Rate', fontsize=12)
axs[1,0].set_title('(d) Logical Error Rate ($d=4$)', fontsize=14)
axs[1,0].grid(True, linestyle='--', alpha=0.6)
axs[1,0].legend()

# ----------------------------
# Subplot 5: Parallel State Space
# ----------------------------
k_values = np.arange(1, 6)
qubit_states = 2 ** k_values
qudit_states = 4 ** k_values
axs[1,1].plot(k_values, qudit_states, 'r-', linewidth=2, label='4D-Qudit')
axs[1,1].plot(k_values, qubit_states, 'b--', linewidth=2, label='2D-Qubit')
axs[1,1].set_xlabel('Number of Quantum Units ($k$)', fontsize=12)
axs[1,1].set_ylabel('Possible States', fontsize=12)
axs[1,1].set_yscale('log')
axs[1,1].set_title('(e) Parallel State Space', fontsize=14)
axs[1,1].grid(True, linestyle='--', alpha=0.6)
axs[1,1].legend()

# ----------------------------
# Subplot 6: Physical Resource Cost
# ----------------------------
qubit_resources = [d**2 for d in code_distances]
qudit_resources = [d for d in code_distances]
axs[1,2].bar(code_distances, qubit_resources, width=0.4, label='2D-Qubit', color='b', alpha=0.6)
axs[1,2].bar(np.array(code_distances)+0.4, qudit_resources, width=0.4, label='4D-Qudit', color='r', alpha=0.6)
axs[1,2].set_xticks(code_distances)
axs[1,2].set_xlabel('Code Distance ($d$)', fontsize=12)
axs[1,2].set_ylabel('Physical Qubits Required', fontsize=12)
axs[1,2].set_title('(f) Logical Qubit Cost', fontsize=14)
axs[1,2].legend()

plt.tight_layout()
plt.savefig('adv_4d.pdf', bbox_inches='tight')
plt.show()