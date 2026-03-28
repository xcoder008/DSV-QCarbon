import numpy as np
import matplotlib.pyplot as plt
import hashlib
import pandas as pd

# 4-dimensional qudit encoding with Gaussian noise model
def encode_4d_qudit(D, t, x, y, H):
    H_part = H[:16]
    bin_H = bin(int(H_part, 16))[2:].zfill(64)
    h_D = int(bin_H[0:16], 2) / (2**16 - 1)
    h_t = int(bin_H[16:32], 2) / (2**16 - 1)
    h_x = int(bin_H[32:48], 2) / (2**16 - 1)
    h_y = int(bin_H[48:64], 2) / (2**16 - 1)
    total = h_D + h_t
    if total > 1:
        h_D /= total
        h_t /= total
    theta_D = (D / 100) * 4 * np.pi
    theta_t = (t / 100) * 2 * np.pi
    theta_x = (x / 100) * np.pi
    theta_y = (y / 100) * np.pi
    a = np.sqrt(h_D) * np.sin(theta_D + theta_x * h_x * 2)
    b = np.sqrt(h_t) * np.cos(theta_t + theta_y * h_y * 2)
    c = np.sqrt(max((1 - h_D - h_t) / 2, 0)) * np.sin(theta_D + theta_t + theta_x + theta_y)
    d = np.sqrt(max(1 - (a**2 + b**2 + c**2), 0))
    return np.array([a, b, c, d])

# Apply Gaussian noise to quantum state
def apply_noise(psi, gamma):
    noise = np.random.normal(0, gamma, 4)
    noisy_psi = psi + noise
    return noisy_psi / np.linalg.norm(noisy_psi)

# Simulate data tampering attacks
def tamper_data(D, t, x, y, attack_type):
    if attack_type == 'none':
        return D, t, x, y
    elif attack_type == 'data':
        return np.random.uniform(0, 100), t, x, y
    elif attack_type == 'st':
        return D, np.random.uniform(0, 100), np.random.uniform(0, 100), np.random.uniform(0, 100)
    elif attack_type == 'both':
        return (np.random.uniform(0, 100), np.random.uniform(0, 100),
                np.random.uniform(0, 100), np.random.uniform(0, 100))

# Calculate state overlap (fidelity-based metric)
def calculate_overlap(psi_ref, psi_obs):
    overlap = np.abs(np.vdot(psi_ref, psi_obs))**2
    return max(1 - overlap, 0)

# Compute dynamic security threshold (95th percentile)
def compute_threshold(noise_probs):
    return np.percentile(noise_probs, 95)

# Run Gaussian noise simulation for threshold calculation
np.random.seed(42)
num_samples = 5000
noise_levels = np.linspace(0.05, 0.3, 6)
attack_types = ['none', 'data', 'st', 'both']

# Generate original reference data
D_orig = np.random.uniform(0, 100, num_samples)
t_orig = np.random.uniform(0, 100, num_samples)
x_orig = np.random.uniform(0, 100, num_samples)
y_orig = np.random.uniform(0, 100, num_samples)

gaussian_tau = []
for gamma in noise_levels:
    collapse_probs_none = []
    for i in range(num_samples):
        H_orig = hashlib.sha256(f"{D_orig[i]}_{t_orig[i]}_{x_orig[i]}_{y_orig[i]}".encode()).hexdigest()
        psi_ref = encode_4d_qudit(D_orig[i], t_orig[i], x_orig[i], y_orig[i], H_orig)
        # Use untampered data to compute threshold
        D_t, t_t, x_t, y_t = tamper_data(D_orig[i], t_orig[i], x_orig[i], y_orig[i], 'none')
        psi_tampered = encode_4d_qudit(D_t, t_t, x_t, y_t, H_orig)
        psi_obs = apply_noise(psi_tampered, gamma)
        prob = calculate_overlap(psi_ref, psi_obs)
        collapse_probs_none.append(prob)
    tau = compute_threshold(collapse_probs_none)
    gaussian_tau.append(tau)
    print(f"Gaussian: γ = {gamma:.2f}, τ = {tau:.4f}")

# Load precomputed results for amplitude damping and phase damping channels
df = pd.read_csv('noise_comparison_results.csv')
amp_data = df[df['Noise Type'] == 'amplitude_damping']
phase_data = df[df['Noise Type'] == 'phase_damping']
amp_data = amp_data.sort_values('γ')
phase_data = phase_data.sort_values('γ')

# Plot dynamic threshold vs noise level
plt.figure(figsize=(8, 6))
plt.plot(noise_levels, gaussian_tau, 'o-', color='#1f77b4', label='Gaussian white noise', linewidth=2, markersize=6)
plt.plot(amp_data['γ'], amp_data['τ'], 's-', color='#ff7f0e', label='Amplitude damping', linewidth=2, markersize=6)
plt.plot(phase_data['γ'], phase_data['τ'], '^-', color='#2ca02c', label='Phase damping', linewidth=2, markersize=6)

plt.xlabel('Noise level $\gamma$', fontsize=12)
plt.ylabel('Dynamic threshold $\tau$', fontsize=12)
plt.title('Adaptive Threshold Adjustment Under Different Noise Channels', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_vs_noise.pdf', dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved as threshold_vs_noise.pdf")