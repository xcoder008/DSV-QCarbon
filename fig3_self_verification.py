import numpy as np
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

def apply_channel_noise(psi, snr_db=30, error_rate=0.01):
    """Apply channel noise to the quantum state psi."""
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1 / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), psi.shape)
    noisy_psi = psi + noise

    for i in range(len(psi)):
        if np.random.rand() < error_rate:
            noisy_psi[i] *= -1 * (1 + np.random.normal(0, 0.05))

    norm = np.linalg.norm(noisy_psi)
    return noisy_psi / norm if norm > 1e-6 else psi

def encode_entangled(D, t, x, y, H):
    """Encode original data into a quantum state using hash H."""
    H_part = H[:16]
    bin_H = bin(int(H_part, 16))[2:].zfill(64)

    h_D = int(bin_H[0:16], 2) / (2 ** 16 - 1)
    h_t = int(bin_H[16:32], 2) / (2 ** 16 - 1)
    h_x = int(bin_H[32:48], 2) / (2 ** 16 - 1)
    h_y = int(bin_H[48:64], 2) / (2 ** 16 - 1)

    total = h_D + h_t
    if total > 1:
        h_D = h_D / total
        h_t = h_t / total

    theta_D = (D / 100) * 4 * np.pi
    theta_t = (t / 100) * 2 * np.pi
    theta_x = (x / 100) * np.pi
    theta_y = (y / 100) * np.pi

    a = np.sqrt(h_D) * np.sin(theta_D + theta_x * h_x * 2)
    b = np.sqrt(h_t) * np.cos(theta_t + theta_y * h_y * 2)
    c = np.sqrt(max((1 - h_D - h_t) / 2, 0)) * np.sin(theta_D + theta_t + theta_x + theta_y)
    d = np.sqrt(max(1 - (a ** 2 + b ** 2 + c ** 2), 0))

    norm = np.linalg.norm([a, b, c, d])
    return np.array([a, b, c, d]) / norm if norm > 1e-6 else np.array([1, 0, 0, 0])

def generate_hash(D, t, x, y):
    """Generate SHA-256 hash based on input data."""
    return hashlib.sha256(f"{D}_{t}_{x}_{y}".encode()).hexdigest()

def self_verification(D_orig, t_orig, x_orig, y_orig,
                      D_tampered, t_tampered, x_tampered, y_tampered,
                      H_tx):
    """Perform self-verification and calculate collapse probability."""
    H_orig = generate_hash(D_orig, t_orig, x_orig, y_orig)
    psi_ref = encode_entangled(D_orig, t_orig, x_orig, y_orig, H_orig)

    psi_tampered = encode_entangled(D_tampered, t_tampered,
                                    x_tampered, y_tampered, H_tx)
    psi_tampered = apply_channel_noise(psi_tampered)

    overlap = np.clip(np.abs(np.vdot(psi_ref, psi_tampered)) ** 2, 0, 1)
    return max(1 - overlap, 0)

num_samples = 5000
np.random.seed(42)

D_orig = np.random.uniform(0, 100, size=num_samples)
t_orig = np.random.uniform(0, 100, size=num_samples)
x_orig = np.random.uniform(0, 100, size=num_samples)
y_orig = np.random.uniform(0, 100, size=num_samples)

indices = np.arange(num_samples)
np.random.shuffle(indices)
group_sizes = [num_samples // 4] * 4
group_sizes[0] += num_samples % 4
start = 0
group_indices = {}
groups = ['No Tampering', 'Data Tampering', 'ST Tampering', 'Data&ST Tampering']
for i, group in enumerate(groups):
    end = start + group_sizes[i]
    group_indices[group] = indices[start:end]
    start = end

D_tampered = D_orig.copy()
t_tampered = t_orig.copy()
x_tampered = x_orig.copy()
y_tampered = y_orig.copy()

for group in groups:
    idxs = group_indices[group]
    for idx in idxs:
        if group == 'Data Tampering':
            D_tampered[idx] = np.random.uniform(0, 100)
        elif group == 'ST Tampering':
            t_tampered[idx] = np.random.uniform(0, 100)
            x_tampered[idx] = np.random.uniform(0, 100)
            y_tampered[idx] = np.random.uniform(0, 100)
        elif group == 'Data&ST Tampering':
            D_tampered[idx] = np.random.uniform(0, 100)
            t_tampered[idx] = np.random.uniform(0, 100)
            x_tampered[idx] = np.random.uniform(0, 100)
            y_tampered[idx] = np.random.uniform(0, 100)

collapse_probs = []
for i in range(num_samples):
    prob = self_verification(
        D_orig[i], t_orig[i], x_orig[i], y_orig[i],
        D_tampered[i], t_tampered[i], x_tampered[i], y_tampered[i],
        generate_hash(D_tampered[i], t_tampered[i], x_tampered[i], y_tampered[i])
    )
    collapse_probs.append(prob)

group_probs = {group: [] for group in groups}
for i in range(num_samples):
    for group in groups:
        if i in group_indices[group]:
            group_probs[group].append(collapse_probs[i])
            break

untouched_probs = group_probs['No Tampering']
tau_candidates = np.percentile(untouched_probs, [99.9, 99.5, 99, 98])

best_tau = None
best_score = -np.inf

for t in tau_candidates:
    far = np.mean([p > t for p in untouched_probs])
    tdr_data = np.mean([p > t for p in group_probs['Data Tampering']])
    tdr_hash = np.mean([p > t for p in group_probs['ST Tampering']])
    tdr_both = np.mean([p > t for p in group_probs['Data&ST Tampering']])

    score = (tdr_data * 2 + tdr_hash + tdr_both) / 4 - 3 * far
    if score > best_score:
        best_tau = t
        best_score = score

tau = max(best_tau, 0.01)
print(f"Dynamic Threshold (τ): {tau:.4f}")

FAR = np.mean([p > tau for p in group_probs['No Tampering']])
print(f"FAR: {FAR:.4f}")

for group in ['Data Tampering', 'ST Tampering', 'Data&ST Tampering']:
    TDR = np.mean([p > tau for p in group_probs[group]])
    print(f"TDR for {group}: {TDR:.4f}")

plt.figure(figsize=(15, 10))
tdr_color = '#2166AC'
normal_bg_color = '#E5F6E3'
alert_bg_color = '#FFEBEB'

for i, group in enumerate(groups, 1):
    plt.subplot(2, 2, i)
    probs_array = np.array(group_probs[group])
    tdr = np.mean(probs_array > tau)

    kde = sns.kdeplot(probs_array, fill=False, label=f"TDR={tdr:.2f}")
    kde_line = kde.lines[0]
    x = kde_line.get_xdata()
    y = kde_line.get_ydata()

    mask = x > tau
    x_above_tau = x[mask]
    y_above_tau = y[mask]
    plt.fill_between(x_above_tau, y_above_tau, alpha=0.3, color=tdr_color)

    plt.axvline(tau, color='red', linestyle='--', label=f"Threshold (τ={tau:.4f})")

    if tdr > 0.5:
        plt.gca().set_facecolor(alert_bg_color)
    else:
        plt.gca().set_facecolor(normal_bg_color)

    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = []
    for handle, label in zip(handles, labels):
        if 'TDR' in label:
            new_handles.append(Patch(facecolor=tdr_color, edgecolor=tdr_color, alpha=0.3, label=label))
        else:
            new_handles.append(handle)

    plt.legend(handles=new_handles)
    plt.xlabel("Collapse Probability")
    plt.ylabel("Density")
    plt.title(f"Group: {group}", fontsize=10)
    plt.xlim(0, 1)

plt.tight_layout()
plt.show()