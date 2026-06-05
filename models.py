"""Core many body model classes for the Lipkin and pairing workflows.

Mostly based on code written by myself and model code from Morten. Codex and
ChatGPT helped organize the classes into a shared model file and make small
improvements.
"""

from __future__ import annotations

import math
from itertools import combinations
import numpy as np


class ManyBodyModel:
    """Minimal base class for models with a Hilbert space dimension and Hamiltonian interface."""
    def __init__(self, dim, name="Model"):
        """Store an integer dimension and display name and returns None."""
        self.dim = dim
        self.name = name

    def hamiltonian(self, g):
        """Accept a scalar coupling and return a Hamiltonian matrix in subclasses."""
        pass

    def spectrum(self, grid, k):
        """Evaluate the lowest k eigenvalues on a coupling grid and return a float array."""
        E = []
        for g in grid:
            H = self.hamiltonian(g)
            evals = np.linalg.eigvalsh(H)
            E.append(evals[:k])
        return np.array(E, dtype=float)


class LipkinModel(ManyBodyModel):
    """Exact Lipkin model and scalar V inputs return Hamiltonian matrices.

    The model formula came from the original code base. Codex and ChatGPT
    mainly helped place it in a reusable class.
    """

    def __init__(self, N, epsilon=1.0, use_symmetric_sector=True):
        """Build Lipkin collective operators from integer N and float epsilon and returns None."""
        self.N = N
        self.epsilon = epsilon
        self.use_symmetric_sector = use_symmetric_sector

        if use_symmetric_sector:
            Jx, Jy, Jz = self._build_collective_operators_symmetric(N)
            dim = Jx.shape[0]
            name = f"Lipkin (symmetric J=N/2, N={N})"
        else:
            Jx, Jy, Jz = self._build_collective_operators_full(N)
            dim = Jx.shape[0]
            name = f"Lipkin (full space, N={N})"

        super().__init__(dim=dim, name=name)

        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz

        self.H0 = epsilon * self.Jz
        self.Hint = (1.0 / N) * (self.Jx @ self.Jx - self.Jy @ self.Jy)

    def hamiltonian(self, V):
        """Evaluate the Lipkin Hamiltonian at scalar V and return a matrix."""
        H = self.H0 + V * self.Hint
        # In the symmetric sector this is real symmetric up to numerical noise.
        return np.real_if_close(H)

    @staticmethod
    def _build_collective_operators_full(N):
        """Build full space collective spin matrices for integer N and return Jx, Jy, Jz."""
        dim = 2**N

        sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        id2 = np.eye(2, dtype=np.complex128)

        Jx = np.zeros((dim, dim), dtype=np.complex128)
        Jy = np.zeros((dim, dim), dtype=np.complex128)
        Jz = np.zeros((dim, dim), dtype=np.complex128)

        for site in range(N):
            ops = []
            for pos in range(N):
                if pos == site:
                    ops.append((sx, sy, sz))
                else:
                    ops.append((id2, id2, id2))

            sx_i, sy_i, sz_i = ops[0]
            for pos in range(1, N):
                sx_i = np.kron(sx_i, ops[pos][0])
                sy_i = np.kron(sy_i, ops[pos][1])
                sz_i = np.kron(sz_i, ops[pos][2])

            Jx += 0.5 * sx_i
            Jy += 0.5 * sy_i
            Jz += 0.5 * sz_i

        return Jx, Jy, Jz

    @staticmethod
    def _build_collective_operators_symmetric(N):
        """Build symmetric sector collective spin matrices for integer N and return Jx, Jy, Jz."""
        J = N / 2.0
        dim = int(2 * J + 1)
        M_vals = np.arange(J, -J - 1, -1, dtype=float)

        Jp = np.zeros((dim, dim), dtype=np.complex128)
        Jm = np.zeros((dim, dim), dtype=np.complex128)
        Jz = np.zeros((dim, dim), dtype=np.complex128)

        for i, M in enumerate(M_vals):
            Jz[i, i] = M

            if i > 0:
                coef = math.sqrt(J * (J + 1.0) - M * (M + 1.0))
                Jp[i - 1, i] = coef

            if i < dim - 1:
                coef = math.sqrt(J * (J + 1.0) - M * (M - 1.0))
                Jm[i + 1, i] = coef

        Jx = 0.5 * (Jp + Jm)
        Jy = -0.5j * (Jp - Jm)

        return Jx, Jy, Jz


class PairingModel(ManyBodyModel):
    """Exact pairing model in the pair occupation basis from pnum, hnum, and delta settings.

    The Hamiltonian logic follows the original notebook implementation. Codex
    and ChatGPT helped organize it into the common model interface.
    """

    def __init__(self, pnum, hnum, delta=1.0):
        """Build the pair occupation basis from pnum, hnum, and delta and returns None."""
        self.pnum = int(pnum)
        self.hnum = int(hnum)
        self.delta = float(delta)

        self.n_levels = (self.hnum + self.pnum) // 2
        self.n_pairs = self.hnum // 2

        self.basis_states = list(combinations(range(self.n_levels), self.n_pairs))
        self.basis_index = {state: idx for idx, state in enumerate(self.basis_states)}

        super().__init__(
            dim=len(self.basis_states),
            name=f"Pairing (pnum={self.pnum}, hnum={self.hnum}, delta={self.delta})",
        )

    def single_particle_energies(self, g):
        """Evaluate pairing single particle energies at scalar g and return a vector."""
        deltaval = 0.5 * self.delta
        gval = -0.5 * float(g)

        epsilon = np.zeros(self.n_levels, dtype=np.float64)
        hole_levels = np.arange(self.n_pairs)
        particle_levels = np.arange(self.pnum // 2)
        epsilon[: self.n_pairs] = deltaval * (2 * hole_levels) + gval
        epsilon[self.n_pairs :] = deltaval * (self.hnum + 2 * particle_levels)
        return epsilon

    def hamiltonian(self, g):
        """Evaluate the exact pairing Hamiltonian at scalar g and return a matrix."""
        g = float(g)
        epsilon = self.single_particle_energies(g)

        H = np.zeros((self.dim, self.dim), dtype=np.float64)

        for i, state_i in enumerate(self.basis_states):
            state_i_set = set(state_i)
            H[i, i] = sum(2.0 * epsilon[k] for k in state_i)

            for l in state_i:
                for k in range(self.n_levels):
                    if k not in state_i_set:
                        state_new = list(state_i)
                        state_new.remove(l)
                        state_new.append(k)
                        state_new = tuple(sorted(state_new))

                        j = self.basis_index[state_new]
                        H[j, i] -= g / 2.0

        return H
