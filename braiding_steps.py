# braiding_steps.py - Standalone QuTiP simulation for Eternal Anyon Braider (TET–CVTL)
# Run: pip install qutip numpy matplotlib
# Then: python braiding_steps.py

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

print("QuTiP versione:", qt.__version__)

# ======================
# PARAMETRI
# ======================
N = 8                   # Qubiti (aumenta se hai RAM)
n_braid_steps = 8
subsystem = [0, 1]

full_dims = [[2] * N, [2] * N]
ket_dims = [[2] * N, [1] * N]

# Stato iniziale composito
zero = qt.basis(2, 0)
psi_current = qt.tensor([zero for _ in range(N)])
psi_current.dims = ket_dims

entropies = []
exp_values = []

print("Inizio simulazione...")

for step in range(n_braid_steps):
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

    braid_op = op1 * op2 * op1.dag()
    braid_op.dims = full_dims

    psi_current = braid_op * psi_current
    psi_current = psi_current.unit()

    rho_sub = psi_current.ptrace(subsystem)
    S = qt.entropy_vn(rho_sub)
    entropies.append(S)

    exp_z = 0.0
    for j in range(N):
        op_list = [qt.qeye(2) for _ in range(N)]
        op_list[j] = qt.sigmaz()
        sz_op = qt.tensor(op_list)
        sz_op.dims = full_dims
        exp_z += qt.expect(sz_op, psi_current)
    exp_z /= N
    exp_values.append(exp_z)

    print(f"Step {step+1}: S = {S:.4f} | <σz>_avg = {exp_z:.4f}")

# PLOTS
if entropies:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(entropies)+1), entropies, 'o-', color='purple')
    plt.xlabel('Passo di Braiding')
    plt.ylabel('Von Neumann Entropy')
    plt.title('Entanglement vs Braid Steps')
    plt.grid(True)
    plt.savefig('entanglement_vs_braid_steps.pdf', dpi=300, bbox_inches='tight')
    print("Salvato: entanglement_vs_braid_steps.pdf")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(exp_values)+1), exp_values, 's-', color='teal')
    plt.xlabel('Passo di Braiding')
    plt.ylabel('<σz> medio')
    plt.title('Expectation σz vs Braid Steps')
    plt.grid(True)
    plt.savefig('exp_sigma_z_vs_braid_steps.pdf', dpi=300, bbox_inches='tight')
    print("Salvato: exp_sigma_z_vs_braid_steps.pdf")

    plt.show()
