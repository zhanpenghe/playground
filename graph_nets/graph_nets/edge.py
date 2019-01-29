
import torch


class EdgeModule(torch.nn.Module):
    """ Base Edge Module.

        Currently edges and connection information are
        represented as a adjacent matrix.
    """
    def __init__(self, connections, self_connected=True):

        assert connections.shape[0] == connections[1]

        super(EdgeModule, self).__init__()
        self.connections = connections
        self.self_connected = self_connected
        self.n_nodes = connections.shape[0]

        if self.self_connected:
            self.adjacent_matrix = self.connections + np.identity(self.n_nodes)

    def forward(self, x):
        """
            The default forward does not do anything.
        """
        return x
