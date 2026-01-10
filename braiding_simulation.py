# Installazione (solo la prima volta)
!pip install qutip -q

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

print("QuTiP versione:", qt.__version__)

# ======================
# PARAMETRI
# ======================
N = 8                   # 8 qubit = 256 dim (veloce)
n_braid_steps = 8
subsystem = [0, 1]      # Entanglement sui primi due qubit

# Dimensione strutturata per sistema composito
full_dims = [[2] * N, [2] * N]          # per operatori
ket_dims = [[2] * N, [1] * N]           # per ket (stati)

# Stato iniziale: prodotto tensore di |0> su ogni qubit
zero = qt.basis(2, 0)
psi_current = qt.tensor([zero for _ in range(N)])
psi_current.dims = ket_dims  # Forza dims composito

entropies = []
exp_values = []

print("Inizio simulazione braiding sequenziale...")

for step in range(n_braid_steps):
    print(f"  Step {step+1}/{n_braid_steps}... ", end="")
    
    # RICREA operatori con dims composito
    op_list1 = [qt.qeye(2) for _ in range(N)]
    op_list1[0] = qt.sigmay()
    op_list1[1] = qt.sigmaz()
    op1 = qt.tensor(op_list1)
    op1.dims = full_dims

    op_list2 = [qt.qeye(2) for _ in range(N)]
    op_list2[0] = qt.sigmaz()
    op_list2[1] = qt.sigmay()
    op2 = qt.tensor(op_list2)
    op2.dims = full_dims

    # Braid cycle placeholder
    braid_op = op1 * op2 * op1.dag()
    braid_op.dims = full_dims

    # Applica braiding
    psi_current = braid_op * psi_current

    # Normalizza
    psi_current = psi_current.unit()

    # Entanglement entropy
    rho_sub = psi_current.ptrace(subsystem)
    S = qt.entropy_vn(rho_sub)
    entropies.append(S)

    # <σz> medio su tutti i qubit
    exp_z = 0.0
    for j in range(N):
        op_list = [qt.qeye(2) for _ in range(N)]
        op_list[j] = qt.sigmaz()
        sz_op = qt.tensor(op_list)
        sz_op.dims = full_dims
        exp_z += qt.expect(sz_op, psi_current)
    exp_z /= N
    exp_values.append(exp_z)

    print(f"S = {S:.4f} | <σz>_avg = {exp_z:.4f}")

# ======================
# PLOTS
# ======================
if entropies:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(entropies)+1), entropies, 'o-', color='purple', lw=2, ms=8)
    plt.xlabel('Passo di Braiding')
    plt.ylabel('Von Neumann Entropy')
    plt.title('Entanglement Entropy vs Braid Steps')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('entanglement_vs_braid_steps.pdf', dpi=300, bbox_inches='tight')
    print("\nPlot salvato: entanglement_vs_braid_steps.pdf")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(exp_values)+1), exp_values, 's-', color='teal', lw=2, ms=8)
    plt.xlabel('Passo di Braiding')
    plt.ylabel('<σz> medio')
    plt.title('Expectation Value σz vs Braid Steps')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp_sigma_z_vs_braid_steps.pdf', dpi=300, bbox_inches='tight')
    print("Plot salvato: exp_sigma_z_vs_braid_steps.pdf")

    plt.show()

    # Stima enhancement
    delta_S = entropies[0] - min(entropies) if len(entropies) > 1 else 0
    fusion_boost = np.exp(delta_S) * 25 if delta_S > 0 else 1
    print(f"\nRiduzione entropia max: ΔS ≈ {delta_S:.4f}")
    print(f"Enhancement stimato fusione: ~{fusion_boost:.1f}×")
else:
    print("Nessun dato.")





