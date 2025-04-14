import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.sparsity_threshold = 0.01

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
                .expand(in_features, -1)
                .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight_real = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight_real = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.base_weight_imag = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight_imag = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight_real, a=math.sqrt(5) * self.scale_base)
        torch.nn.init.kaiming_uniform_(self.base_weight_imag, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight_real.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            self.spline_weight_imag.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def b_splines_complex(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given complex input tensor.

        Args:
            x (torch.Tensor): Complex input tensor of shape (batch_size, in_features), where x contains both real and imaginary parts.

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # Split real and imaginary parts
        real_x = x.real.unsqueeze(-1)  # Real part of x
        imag_x = x.imag.unsqueeze(-1)  # Imaginary part of x

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)

        # Zero order B-spline for real and imaginary parts
        bases_real = ((real_x >= grid[:, :-1]) & (real_x < grid[:, 1:])).to(real_x.dtype)
        bases_imag = ((imag_x >= grid[:, :-1]) & (imag_x < grid[:, 1:])).to(imag_x.dtype)

        # Recursion for higher order B-splines
        for k in range(1, self.spline_order + 1):
            # Compute real part for the recursive terms
            real_term_1 = ((real_x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]))
            real_term_2 = ((grid[:, k + 1:] - real_x) / (grid[:, k + 1:] - grid[:, 1:(-k)]))

            # Compute imaginary part for the recursive terms
            imag_term_1 = ((imag_x - 0) / (grid[:, k:-1] - grid[:, : -(k + 1)]))
            imag_term_2 = ((0 - imag_x) / (grid[:, k + 1:] - grid[:, 1:(-k)]))

            # Combine real and imaginary contributions
            real1 = real_term_1 * bases_real[:, :, :-1] - (imag_term_1 * bases_imag[:, :, :-1])
            imag1 = imag_term_1 * bases_real[:, :, :-1] + (real_term_1 * bases_imag[:, :, :-1])

            real2 = real_term_2 * bases_real[:, :, 1:] - (imag_term_2 * bases_imag[:, :, 1:])
            imag2 = imag_term_2 * bases_real[:, :, 1:] + (real_term_2 * bases_imag[:, :, 1:])

            # Update real and imaginary parts separately
            bases_real = real1 + real2
            bases_imag = imag1 + imag2

        assert bases_real.size() == (
            real_x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )

        result = torch.stack([bases_real, bases_imag], dim=-1)
        result = torch.view_as_complex(result)

        return result.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        # print(x.shape, y.shape)
        # print(y.size())
        # print(x.size(0))
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0,1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight_real(self):
        return self.spline_weight_real * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    @property
    def scaled_spline_weight_imag(self):
        return self.spline_weight_imag * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def complex_silu(self, x_real: torch.Tensor, x_imag: torch.Tensor):
        exp_neg_real = torch.exp(-x_real)
        cos_imag = torch.cos(x_imag)
        sin_imag = torch.sin(x_imag)

        denominator_real = 1 + exp_neg_real * cos_imag
        denominator_imag = exp_neg_real * sin_imag

        denom_magnitude_squared = denominator_real ** 2 + denominator_imag ** 2

        real_part = (x_real * denominator_real - x_imag * denominator_imag) / denom_magnitude_squared
        imag_part = (x_imag * denominator_real + x_real * denominator_imag) / denom_magnitude_squared

        return real_part, imag_part

    def forward(self, x: torch.Tensor):
        # assert x.size(-1) == self.in_features
        original_shape = x.shape

        B, N, T, dimension = x.shape

        # x_real = x.real.view(B * nt, dimension)
        # x_imag = x.imag.view(B * nt, dimension)

        x = x.reshape(-1, self.in_features)
        x_real = x.real.reshape(-1, self.in_features)
        x_imag = x.imag.reshape(-1, self.in_features)

        silu_real, silu_imag = self.complex_silu(x_real, x_imag)

        bspline = self.b_splines_complex(x)
        bspline_real = bspline.real
        bspline_imag = bspline.imag

        base_output_real = F.linear(silu_real, self.base_weight_real) \
                           - F.linear(silu_imag, self.base_weight_imag)

        spline_output_real = F.linear(
            bspline_real.view(x_real.size(0), -1),
            self.scaled_spline_weight_real.view(self.out_features, -1),
        ) - F.linear(
            bspline_imag.view(x_imag.size(0), -1),
            self.scaled_spline_weight_imag.view(self.out_features, -1),
        )
        # spline_output_real = F.linear(
        #     self.b_splines(x_real).view(x_real.size(0), -1),
        #     self.scaled_spline_weight_real.view(self.out_features, -1),
        # ) - F.linear(
        #     self.b_splines(x_imag).view(x_imag.size(0), -1),
        #     self.scaled_spline_weight_imag.view(self.out_features, -1),
        # )
        output_real = base_output_real + spline_output_real

        base_output_imag = F.linear(silu_imag, self.base_weight_real) \
                           + F.linear(silu_real, self.base_weight_imag)

        spline_output_imag = F.linear(
            bspline_imag.view(x_imag.size(0), -1),
            self.scaled_spline_weight_real.view(self.out_features, -1),
        ) + F.linear(
            bspline_real.view(x_real.size(0), -1),
            self.scaled_spline_weight_imag.view(self.out_features, -1),
        )
        # spline_output_imag = F.linear(
        #     self.b_splines(x_imag).view(x_imag.size(0), -1),
        #     self.scaled_spline_weight_real.view(self.out_features, -1),
        # ) + F.linear(
        #     self.b_splines(x_real).view(x_real.size(0), -1),
        #     self.scaled_spline_weight_imag.view(self.out_features, -1),
        # )
        output_imag = base_output_imag + spline_output_imag

        output = torch.stack([output_real, output_imag], dim=-1)
        output = F.softshrink(output, lambd=self.sparsity_threshold)
        output = torch.view_as_complex(output)

        output = output.reshape(*original_shape[:-1], self.out_features)

        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        # assert x.dim() == 2 and x.size(-1) == self.in_features
        assert x.size(-1) == self.in_features
        x = x.reshape(-1, self.in_features)
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight_real  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x.real, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight_real.data.copy_(self.curve2coeff(x.real, unreduced_spline_output))
        self.spline_weight_imag.data.copy_(self.curve2coeff(x.imag, unreduced_spline_output))

    # @torch.no_grad()
    # def update_grid(self, x: torch.Tensor, margin=0.01):
    #     assert x.size(-1) == self.in_features
    #     # assert x.dim() == 2 and x.size(1) == self.in_features
    #     x = x.reshape(-1, self.in_features)
    #     batch = x.size(0)
    #
    #     x_real = x.real
    #     x_imag = x.imag
    #
    #     splines_real = self.b_splines(x_real)  # (batch, in_features, coeff)
    #     splines_imag = self.b_splines(x_imag)  # (batch, in_features, coeff)
    #
    #     splines_real = splines_real.permute(1, 0, 2)  # (in_features, batch, coeff)
    #     splines_imag = splines_imag.permute(1, 0, 2)  # (in_features, batch, coeff)
    #
    #     orig_coeff_real = self.scaled_spline_weight_real  # (out_features, in_features, coeff)
    #     orig_coeff_imag = self.scaled_spline_weight_imag  # (out_features, in_features, coeff)
    #
    #     orig_coeff_real = orig_coeff_real.permute(1, 2, 0)  # (in_features, coeff, out_features)
    #     orig_coeff_imag = orig_coeff_imag.permute(1, 2, 0)  # (in_features, coeff, out_features)
    #
    #     unreduced_spline_output_real = torch.bmm(splines_real, orig_coeff_real)  # (in_features, batch, out_features)
    #     unreduced_spline_output_imag = torch.bmm(splines_imag, orig_coeff_imag)  # (in_features, batch, out_features)
    #
    #     # sort each channel individually to collect data distribution
    #     x_real_sorted = torch.sort(x_real, dim=0)[0]
    #     x_imag_sorted = torch.sort(x_imag, dim=0)[0]
    #     grid_adaptive = x_real_sorted[
    #         torch.linspace(
    #             0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
    #         )
    #     ]
    #
    #     uniform_step = (x_real_sorted[-1] - x_real_sorted[0] + 2 * margin) / self.grid_size
    #     grid_uniform = (
    #             torch.arange(
    #                 self.grid_size + 1, dtype=torch.float32, device=x.device
    #             ).unsqueeze(1)
    #             * uniform_step
    #             + x_real_sorted[0]
    #             - margin
    #     )
    #
    #     grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
    #     grid = torch.cat(
    #         [
    #             grid[:1]
    #             - uniform_step
    #             * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
    #             grid,
    #             grid[-1:]
    #             + uniform_step
    #             * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
    #         ],
    #         dim=0,
    #     )
    #
    #     self.grid.copy_(grid.T)
    #     self.spline_weight_real.data.copy_(self.curve2coeff(x_real, unreduced_spline_output_real.transpose(0, 1)))
    #     self.spline_weight_imag.data.copy_(self.curve2coeff(x_imag, unreduced_spline_output_imag.transpose(0, 1)))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_real = self.spline_weight_real.abs().mean(-1)
        l1_imag = self.spline_weight_imag.abs().mean(-1)

        regularization_loss_activation_real = l1_real.sum()
        regularization_loss_activation_imag = l1_imag.sum()
        regularization_loss_activation = regularization_loss_activation_real + regularization_loss_activation_imag

        p_real = l1_real / regularization_loss_activation_real
        p_imag = l1_imag / regularization_loss_activation_imag

        regularization_loss_entropy_real = -torch.sum(p_real * p_real.log())
        regularization_loss_entropy_imag = -torch.sum(p_imag * p_imag.log())
        regularization_loss_entropy = regularization_loss_entropy_real + regularization_loss_entropy_imag

        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            regularize_activation=1.0,
            regularize_entropy=1.0,
            update_grid=True,
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.regularize_activation = regularize_activation
        self.regularize_entropy = regularize_entropy

        self.update_grid = update_grid

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            if self.update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self):
        return sum(
            layer.regularization_loss(self.regularize_activation, self.regularize_entropy)
            for layer in self.layers
        )
