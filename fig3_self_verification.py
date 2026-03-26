import numpy as np
import matplotlib.pyplot as plt
import hashlib


# --- 4D Qudit Quantum State Encoding ---
def encode_4d_qudit(D, t, x, y, H):
    H_part = H[:16]
    bin_H = bin(int(H_part, 16))[2:].zfill(64)

    h_D = int(bin_H[0:16], 2) / (2 ** 16 - 1)
    h_t = int(bin_H[16:32], 2) / (2 ** 16 - 1)
    h_x = int(bin_H[32:48], 2) / (2 ** 16 - 1)
    h_y = int(bin_H[48:64], 2) / (2 ** 16 - 1)

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
    d = np.sqrt(max(1 - (a ** 2 + b ** 2 + c ** 2), 0))

    return np.array([a, b, c, d])


# --- Quantum Channel Noise ---
def apply_noise(psi, gamma):
    noise = np.random.normal(0, gamma, 4)
    noisy_psi = psi + noise
    return noisy_psi / np.linalg.norm(noisy_psi)


# --- Data Tampering Simulation ---
def tamper_data(D, t, x, y, attack_type):
    if attack_type == 'none':
        return D, t, x, y
    elif attack_type == 'data':
        return np.random.uniform(0, 100), t, x, y
    elif attack_type == 'st':
        return D, np.random.uniform(0, 100), np.random.uniform(0, 100), np.random.uniform(0, 100)
    elif attack_type == 'both':
        return np.random.uniform(0, 100), np.random.uniform(0, 100), np.random.uniform(0, 100), np.random.uniform(0, 100)


# --- Collapse Probability Calculation ---
def calculate_overlap(psi_ref, psi_obs):
    overlap = np.abs(np.dot(psi_ref, psi_obs)) ** 2
    return max(1 - overlap, 0)


# --- Dynamic Threshold Calculation ---
def compute_threshold(noise_probs):
    """
    Compute 95th percentile to ensure FAR ≤ 5%
    """
    return np.percentile(noise_probs, 95)


# --- Experiment Configuration ---
np.random.seed(42)
num_samples = 5000
noise_levels = np.linspace(0.05, 0.3, 6)
attack_types = ['none', 'data', 'st', 'both']

results = {gamma: {} for gamma in noise_levels}

D_orig = np.random.uniform(0, 100, num_samples)
t_orig = np.random.uniform(0, 100, num_samples)
x_orig = np.random.uniform(0, 100, num_samples)
y_orig = np.random.uniform(0, 100, num_samples)

for gamma in noise_levels:
    print(f"Processing γ = {gamma:.2f}")
    collapse_probs = {attack: [] for attack in attack_types}

    for i in range(num_samples):
        H_orig = hashlib.sha256(f"{D_orig[i]}_{t_orig[i]}_{x_orig[i]}_{y_orig[i]}".encode()).hexdigest()
        psi_ref = encode_4d_qudit(D_orig[i], t_orig[i], x_orig[i], y_orig[i], H_orig)

        for attack in attack_types:
            D_t, t_t, x_t, y_t = tamper_data(D_orig[i], t_orig[i], x_orig[i], y_orig[i], attack)
            psi_tampered = encode_4d_qudit(D_t, t_t, x_t, y_t, H_orig)
            psi_obs = apply_noise(psi_tampered, gamma)
            prob = calculate_overlap(psi_ref, psi_obs)
            collapse_probs[attack].append(prob)

    tau = compute_threshold(collapse_probs['none'])
    results[gamma]['tau'] = tau
    results[gamma]['FAR'] = np.mean(collapse_probs['none'] > tau)
    results[gamma]['TDR'] = {attack: np.mean(collapse_probs[attack] > tau)
                             for attack in attack_types if attack != 'none'}
    results[gamma]['probs'] = collapse_probs

# --- Print All Results ---
print("=== Simulation Results ===")
for gamma in noise_levels:
    print(f"\nNoise Level γ = {gamma:.2f}:")
    print(f"  Dynamic Threshold τ = {results[gamma]['tau']:.4f}")
    print(f"  FAR = {results[gamma]['FAR'] * 100:.1f}%")
    print("  TDR:")
    for attack, tdr in results[gamma]['TDR'].items():
        print(f"    {attack:<10}: {tdr * 100:.1f}%")

# --- Visualization ---
attack_info = {
    'data': 'Data Tampering',
    'st': 'Spatiotemporal Tampering',
    'both': 'Data & ST Tampering'
}

for attack in ['data', 'st', 'both']:
    plt.figure(figsize=(18, 5))

    for i, gamma in enumerate(noise_levels, 1):
        plt.subplot(2, 3, i)
        probs_tampered = results[gamma]['probs'][attack]
        probs_untampered = results[gamma]['probs']['none']
        tau = results[gamma]['tau']

        plt.hist(probs_tampered, bins=30, density=True, alpha=0.5, color='red', label='Tampered')
        plt.hist(probs_untampered, bins=30, density=True, alpha=0.4, color='#003366', label='Untampered')
        plt.axvline(tau, color='black', linestyle='--', label=f'τ = {tau:.2f}')
        plt.axvline(0.5, color='gray', linestyle=':', label='Static Threshold')

        plt.text(0.46, 0.9, f"γ = {gamma:.2f}", transform=plt.gca().transAxes)
        plt.xlabel('Detection Probability')
        plt.ylabel('Density')
        plt.legend()
        plt.xlim(0, 1)

    plt.tight_layout()
    plt.show()

# --- Dynamic Threshold Trend ---
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, [results[g]['tau'] for g in noise_levels], 'ro-', label='Dynamic Threshold (τ)')
plt.xlabel('Noise Level (γ)')
plt.ylabel('Threshold Value (τ)')
plt.title('Dynamic Threshold vs Noise Level')
plt.grid(True)
plt.show()
