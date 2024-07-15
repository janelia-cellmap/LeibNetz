# %%
from typing import Callable, Sequence
import torch
import numpy as np
from tqdm import trange


class Basis(torch.nn.Module):
    """Basis. This is the base class for all basis functions."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def new(cls, *args, **kwargs) -> "Basis":
        return cls(*args, **kwargs)


class WaveletBasis(Basis):
    """Wavelet basis. This parameterizes a basis as a linear combination of wavelet basis functions.

    Args:
        num_wavelets (int): Number of wavelets.

    Attributes:
        num_wavelets (int): Number of wavelets.
        coefficients (torch.nn.Parameter): Coefficients of the wavelet basis.
    """

    def __init__(self, num_wavelets=27):
        super().__init__()
        self.num_wavelets = num_wavelets
        ...
        # TODO: Implement wavelet basis
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.coefficients, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width, [depth])
        # coefficients: (num_wavelets*2 + 1)
        # out: (batch_size, channels, height, width, [depth])
        out = torch.zeros_like(x)
        ...
        # TODO: Implement wavelet basis
        return out


class PoloynomialBasis(Basis):
    """Polynomial basis. This parameterizes a basis as a linear combination of polynomial basis functions.

    Args:
        num_degrees (int): Number of degrees.

    Attributes:
        num_degrees (int): Number of degrees.
        coefficients (torch.nn.Parameter): Coefficients of the polynomial basis.
    """

    def __init__(self, num_degrees=27):
        super().__init__()
        self.num_degrees = num_degrees
        self.coefficients = torch.nn.Parameter(
            torch.Tensor(num_degrees + 1), requires_grad=True
        )
        function = lambda x: self.coefficients[0]
        for i in range(1, num_degrees + 1):
            function = lambda x: function(x) + self.coefficients[i] * x**i
        self.function = function
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.coefficients, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width, [depth])
        # coefficients: (num_degrees + 1)
        # out: (batch_size, channels, height, width, [depth])
        out = self.function(x)
        return out


class FourierBasis(Basis):
    """Fourier basis. This parameterizes a basis as a linear combination of Fourier basis functions.

    Args:
        num_frequencies (int): Number of frequencies.

    Attributes:
        num_frequencies (int): Number of frequencies.
        coefficients (torch.nn.Parameter): Coefficients of the Fourier basis.
    """

    def __init__(self, num_frequencies=27):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.coefficients = torch.nn.Parameter(
            torch.Tensor(2 * num_frequencies + 1), requires_grad=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.coefficients, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width, [depth])
        # coefficients: (num_frequencies*2 + 1)
        # out: (batch_size, channels, height, width, [depth])
        out = torch.zeros_like(x)
        for i in torch.arange(self.num_frequencies):
            out += self.coefficients[2 * i] * torch.cos(2 * torch.pi * i * x)
            out += self.coefficients[2 * i + 1] * torch.sin(2 * torch.pi * i * x)
        out += self.coefficients[2 * self.num_frequencies]
        return out


class BSplineBasis(Basis):
    """B-spline basis. This parameterizes a basis as a linear combination of B-spline basis functions.

    Args:
        num_knots (int): Number of knots.

    Attributes:
        num_knots (int): Number of knots.
        coefficients (torch.nn.Parameter): Coefficients of the B-spline basis.
    """

    def __init__(self, num_knots=27):
        super().__init__()
        self.num_knots = num_knots
        raise NotImplementedError("B-spline basis is not implemented yet.")


class Kernel(torch.nn.Module):
    """Kernel. This is the base class for all kernels.

    Args:
        basis (Basis): Basis function.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Sequence[int]): Kernel size.

    Attributes:
        basis (Basis): Basis function.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Sequence[int]): Kernel size.
        kernel_functions (torch.nn.ModuleDict): Kernel functions.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Forward pass.
    """

    def __init__(
        self,
        basis: Basis,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int] = [3, 3, 3],
        # TODO: Add "same" padding
    ):
        super().__init__()
        self.basis = basis
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_shape = [out_channels, in_channels] + kernel_size
        self.kernel = np.ndarray(self.kernel_shape, dtype=object)
        for index in np.ndindex(tuple(self.kernel_shape)):
            self.kernel[index] = basis.new()
        self.kernel_functions = torch.nn.ModuleList(self.kernel.flatten())
        slices = []
        for size in self.kernel_shape[2:]:
            these_slices = []
            for i in range(size):
                start = i
                end = -(size - 1) + i if i != size - 1 else None
                these_slices.append(slice(start, end))
            slices.append(these_slices)
        self.slices = slices

    def __matmul__(self, x):
        return self.forward(x)

    def _recurse_kernel(
        self, x: torch.Tensor, out_channel: int, in_channel: int, inds: list, out
    ) -> torch.Tensor:
        if isinstance(self.kernel[out_channel, in_channel, *inds], Basis):
            out[:, out_channel, ...] += self.kernel[out_channel, in_channel, *inds](  # type: ignore
                x[
                    :,
                    in_channel,
                    *[slices[ind] for ind, slices in zip(inds, self.slices)],
                ]
            )
        else:
            for i in range(len(self.kernel[out_channel, in_channel, *inds])):
                inds.append(i)
                out = self._recurse_kernel(x, out_channel, in_channel, inds, out)
                inds.pop()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, height, width, [depth])
        # out: (batch_size, out_channels, height - (kernel_size-1), width - (kernel_size-1), [depth - (kernel_size-1)])
        padding = np.array(self.kernel_size) - 1
        out_spatial_shape = np.array(x.shape[2:]) - padding
        out_shape = [x.shape[0], self.out_channels, *out_spatial_shape]
        out = torch.zeros(out_shape, device=x.device)
        for in_channel in range(self.kernel_shape[1]):
            for out_channel in range(self.kernel_shape[0]):
                out = self._recurse_kernel(x, out_channel, in_channel, [], out)
        return out


class BasisModel(torch.nn.Module):
    # TODO: Add needed parameters/functions to be a LeibNet node
    """BasisModel. This is the base class for all basis models.

    Args:
        kernels (Kernel | Sequence[Kernel]): Kernels.
        reduce_function (Callable | None): Reduce function.

    Attributes:
        kernels (torch.nn.ModuleList): Kernels.
        reduce_function (Callable | None): Reduce function.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Forward pass.
    """

    def __init__(
        self,
        kernels: Kernel | Sequence[Kernel],
        reduce_function: Callable | None = None,
    ):
        super().__init__()
        if isinstance(kernels, Sequence):
            self.kernels = torch.nn.Sequential(*kernels)
        else:
            self.kernels = kernels
        self.reduce_function = reduce_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, height, width, [depth])
        # out: (batch_size, out_channels, height, width, [depth])
        out = self.kernels(x)
        if self.reduce_function is not None:
            out = self.reduce_function(out)
        return out
