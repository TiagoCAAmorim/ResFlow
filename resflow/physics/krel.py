"""Relative permeability model (tables + Stone II).

Copyright (c) Tiago Amorim. MIT License.
"""
import torch
from torch import nn

from .interp import interp1d, search_interval

EPS = 1e-20


class KRELModel(nn.Module):
    """Relative permeability from oil-water and gas-oil tables (Stone II for three phases).

    Args:
        wo_table: (n, 3) oil-water table with columns Sw, Krw, Krow.
        go_table: (n, 3) gas-oil table with columns Sg, Krg, Krog.

    The tables are stored transposed (`[3, n]`) in the buffers `wo_table` and `go_table`.
    """

    def __init__(self, wo_table: torch.Tensor, go_table: torch.Tensor):
        super().__init__()
        self.register_buffer('wo_table', wo_table.T.contiguous())
        self.register_buffer('go_table', go_table.T.contiguous())

        self.sg_crit = self.go_table[0][self.go_table[1] == 0].max()
        self.sg_quad = self.go_table[0][self.go_table[1] > 0].min()
        self.kro_max = self.go_table[2].max()
        # Plain float for the clamp in get_kr (a tensor bound would be read back to the host).
        self._kro_max_f = float(self.kro_max)

        if torch.abs(self.wo_table[2].max() - self.go_table[2].max()) > EPS:
            print("Warning: Kro max do not coincide in wo and go tables.")

    def get_krw(self, sw: torch.Tensor, return_o: bool = False,
                idx: torch.Tensor = None) -> torch.Tensor:
        """Water relative permeability (or oil-water Krow if `return_o`) at water saturation sw.

        `idx` is an optional precomputed search_interval(wo_table[0], sw.clamp(0, 1)).
        """
        sw = torch.clamp(sw, 0.0, 1.0)
        i = 2 if return_o else 1
        return interp1d(self.wo_table[0], self.wo_table[i], sw, idx=idx)

    def get_krg(self, sg: torch.Tensor, return_o: bool = False) -> torch.Tensor:
        """Gas relative permeability (or gas-oil Krog if `return_o`) at gas saturation sg.

        Below the first point with Krg > 0, Krg follows a quadratic start (smooth onset).
        """
        sg = torch.clamp(sg, 0.0, 1.0)

        if return_o:
            i = 2
        else:
            i = 1
            sg_quad = (sg - self.sg_crit) ** 2 / (self.sg_quad - self.sg_crit) + self.sg_crit
            sg = torch.where(sg > self.sg_quad, sg, sg_quad)

        return interp1d(self.go_table[0], self.go_table[i], sg)

    def get_kr(self, sw: torch.Tensor, sg: torch.Tensor) -> torch.Tensor:
        """Oil, gas and water relative permeability (Stone II). Returns [..., 3]: kro, krg, krw."""
        idx_w = search_interval(self.wo_table[0], torch.clamp(sw, 0.0, 1.0))
        krw = self.get_krw(sw, idx=idx_w)
        krg = self.get_krg(sg)
        krow = self.get_krw(sw, return_o=True, idx=idx_w)
        krog = self.get_krg(sg, return_o=True)
        kro_cw = self.kro_max

        kro = kro_cw * ((krow / kro_cw + krw) * (krog / kro_cw + krg) - krw - krg)
        kro = torch.clamp(kro, min=0.0, max=self._kro_max_f)
        return torch.stack((kro, krg, krw), dim=-1)

    def get(self, sw: torch.Tensor, sg: torch.Tensor) -> torch.Tensor:
        """Alias of get_kr."""
        return self.get_kr(sw, sg)

    def forward(self, x):
        """x: [..., 2] with columns (sw, sg). Returns [..., 3]: kro, krg, krw."""
        return self.get(x[:, 0], x[:, 1])
