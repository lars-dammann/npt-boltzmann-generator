"""Define the equivariant graph neural network"""

import torch


class EGNN(torch.nn.Module):
    """Equivariant graph neural network"""

    def __init__(
        self,
        n_dim,
        n_volume,
        n_particles,
        hidden_nf,
        n_layers,
        attention,
        norm_constant,
        edge_feat_nf=1,
        act_fn=torch.nn.SiLU(),
        init_weights_std=1.0,
        init_weights_att_gain=1.0,
    ):
        super().__init__()

        self.n_dim = n_dim
        self.n_volume = n_volume
        self.n_particles = n_particles
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.embedding = torch.nn.Linear(
            self.n_particles, self.n_particles * self.hidden_nf
        )

        for i in range(0, n_layers):
            if i == (self.n_layers - 1):
                self.add_module(
                    f"e_block_{i}",
                    EquivariantBlock(
                        hidden_nf=hidden_nf,
                        edge_feat_nf=edge_feat_nf,
                        n_dim=n_dim,
                        n_particles=n_particles,
                        act_fn=act_fn,
                        attention=attention,
                        norm_constant=norm_constant,
                        last=True,
                    ),
                )
            else:
                self.add_module(
                    "e_block_{i}",
                    EquivariantBlock(
                        hidden_nf=hidden_nf,
                        edge_feat_nf=edge_feat_nf,
                        n_dim=n_dim,
                        n_particles=n_particles,
                        act_fn=act_fn,
                        attention=attention,
                        norm_constant=norm_constant,
                        last=False,
                    ),
                )

        self._normal_init(self, std=init_weights_std)
        self._att_init(self, gain=init_weights_att_gain)

    def _normal_init(self, model, std=1.0):
        for name, param in model.named_parameters():
            param.data.normal_(mean=0.0, std=std)

    def _att_init(self, model, gain=1.0):
        for name, param in model.named_parameters():
            if "edge_att_mlp" in name:
                torch.nn.init.xavier_uniform_(param.data, gain=gain)

    def forward(self, x_inp):
        """Forward pass of EGNN"""
        x = x_inp.clone()

        if x.shape[0] != (self.n_particles * self.n_dim):
            raise ValueError("Wrong formatting of data")

        x = x.reshape(self.n_particles, self.n_dim)

        with torch.inference_mode(False):
            h = torch.ones(x.shape[0]).to(x)
        h = self.embedding(h).reshape(x.shape[0], -1)

        for i in range(0, self.n_layers):
            if i == (self.n_layers - 1):
                x = self._modules[f"e_block_{i}"](x, h, last=True)
            else:
                x, h = self._modules[f"e_block_{i}"](x, h, last=False)

        try:
            if torch.any(torch.isnan(x)):
                print("Warning: detected nan, resetting EGNN output to zero.")
                x = torch.zeros_like(x)
        except RuntimeError:
            pass

        return x.reshape(self.n_particles * self.n_dim)


class EquivariantBlock(torch.nn.Module):
    """One layer of equivariant transformation"""

    def __init__(
        self,
        n_dim,
        n_particles,
        hidden_nf,
        edge_feat_nf=1,
        act_fn=torch.nn.SiLU(),
        attention=True,
        norm_constant=1,
        last=False,
    ):
        super(EquivariantBlock, self).__init__()
        input_nf = hidden_nf
        output_nf = hidden_nf
        input_edge = hidden_nf * 2 + edge_feat_nf
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.norm_constant = norm_constant
        self.attention = attention

        if not last:
            self.edge_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_edge, hidden_nf),
                act_fn,
                torch.nn.Linear(hidden_nf, hidden_nf),
                act_fn,
            )

            self.node_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_nf + input_nf, hidden_nf),
                act_fn,
                torch.nn.Linear(hidden_nf, output_nf),
            )

            if self.attention:
                self.edge_att_mlp = torch.nn.Sequential(
                    torch.nn.Linear(hidden_nf, 1, bias=False), torch.nn.Sigmoid()
                )

        # Note that this definition is not the same as in the paper of Satorras
        self.coord_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_edge, hidden_nf),
            act_fn,
            torch.nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            torch.nn.Linear(hidden_nf, 1),
        )

        # Create indices to slice through coordinate distances
        with torch.inference_mode(False):
            # Combine all all of diagonal indices of a matrix into a tensor
            indices = torch.cat(
                [
                    torch.triu_indices(n_particles, n_particles, offset=1).t(),
                    torch.tril_indices(n_particles, n_particles, offset=-1).t(),
                ]
            )
            # Sort the indices into tensor after first index,
            # so one gets a tensor of size n_particels x (n_particels - 1) that is easily sliceable
            self.indices = (
                indices[indices[:, 0].argsort()]
                .reshape(n_particles, n_particles - 1, 2)
                .contiguous()
            )

    def forward(self, x, h, last):
        """Forward pass through one layer of EGNN"""
        # Calculate edges m_ij and nodes h
        coord_diff = self.calc_coordinate_diff(x)
        sqrd_dist = self.calc_sqrd_dist(coord_diff)
        x = self.calc_coord(h, x, coord_diff, sqrd_dist)
        if not last:
            h = self.calc_nodes(h, sqrd_dist)
            return x, h
        else:
            return x

    def calc_coordinate_diff(self, x):
        """Calculates the coordinate difference of all atoms,
         excluding duplicates and self interactions"""
        return x[self.indices].diff(dim=2).squeeze(dim=2)

    def calc_sqrd_dist(self, coord_diff):
        """Calculates the minimum image squared distances"""
        # TODO:check
        return torch.sum(torch.pow(coord_diff, 2), dim=2, keepdim=True)

    def calc_coord(self, h, x, coord_diff, sqrd_dist):
        """Calculate forward pass of atom coordinates through EGNN layer"""
        mlp_input = torch.cat(
            [
                h[self.indices[:, :, 0].flatten()],
                h[self.indices[:, :, 1].flatten()],
                sqrd_dist.reshape(-1, 1),
            ],
            dim=1,
        )
        # Calculate updated coordinates x_i
        x_ij = self.calc_coordinate_weight(coord_diff, sqrd_dist) * self.coord_mlp(
            mlp_input
        ).reshape(self.indices.shape[0], self.indices.shape[1], 1)
        # Take every atom i and update with sum_j x_ij
        x = x + torch.sum(x_ij, dim=1)
        return x

    def calc_coordinate_weight(self, coord_diff, sqrd_dist, constant=1):
        """Calculates the individual weighting per atom atom pair for the coordinate update"""
        return coord_diff / (torch.sqrt(sqrd_dist) + constant)

    def calc_nodes(self, h, sqrd_dist):
        """Calculate forward pass of nodes through EGNN"""
        mlp_input = torch.cat(
            [
                h[self.indices[:, :, 0].flatten()],
                h[self.indices[:, :, 1].flatten()],
                sqrd_dist.reshape(-1, 1),
            ],
            dim=1,
        )
        m_ij = self.edge_mlp(mlp_input)
        if self.attention:
            m_ij = m_ij * self.edge_att_mlp(m_ij)
        # Reshape back to n x n-1 matrix to sum over j
        m_ij = m_ij.reshape(self.indices.shape[0], self.indices.shape[1], -1)
        return self.node_mlp(torch.cat((h, torch.sum(m_ij, dim=1)), dim=1))
