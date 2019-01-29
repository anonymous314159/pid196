import math
import torch
from torch.autograd import Variable
from trainer import Trainer
from utils import AverageMeter


class OrthReguTrainer(Trainer):
    """
    This class contains all core code of our proposed orthogonality based on random SVD.
    """
    def __init__(self, model, criterion, optimizer, args):
        super().__init__(model, criterion, optimizer, args)
        self._k_coeff = 3
        self._q = 1
        self._bsize = 0
        self._ortho_decay = args.ortho_decay
        self._init_ortho_decay = args.ortho_decay

    def before_train(self, epoch, epochs):
        self.adjust_ortho_decay_rate(epoch + 1, epochs)

    def adjust_ortho_decay_rate(self, epoch, num_epochs):
        coeff = 0.5 * (1 + math.cos(epoch * math.pi / num_epochs))
        self._ortho_decay = coeff * self._init_ortho_decay
            
    def _backward(self, batch_loss, epoch, update=True, retain_graph=False):
        self._optimizer.zero_grad()
        batch_loss.backward(retain_graph=retain_graph)
        if update:
            self.update_gradient()
            self._optimizer.step()

    def update_gradient(self):
        """
        The gradient from our orthogonality regularization is directly attached to
        the gradient from the normal back-propagation process.
        :return:
        """
        for W in self._model.parameters():
            cols = W[0].numel()
            Wl = W.view(-1, cols)
            try:
                grad = self.compute_graident(Wl)
                W.grad.data += self._ortho_decay * grad.reshape_as(W.grad.data)
            except Exception as e:
                print(str(e))

    def random_svd(self, A, k):
        if self._bsize == 0:
            bsize = k
        u = A.new_zeros((1, A.shape[1]))
        l = A.new_zeros((A.shape[0], 1))

        if A.shape[0] < A.shape[1]:
            n = A.shape[0]
            ind = 0
        else:
            n = A.shape[1]
            ind = 1
        tpose = False

        if ind == 0:
            tpose = True
            l = torch.t(u)
            u = A.new_ones((1, A.shape[0]))
            A = torch.t(A)
        K = A.new_zeros((A.shape[1], bsize * self._q))
        block = torch.randn(A.shape[1], bsize).to(A.device)
        block, _ = torch.qr(block)
        T = A.new_zeros((A.shape[1], bsize))

        for i in range(self._q):
            T = torch.matmul(A, block) - torch.matmul(l, torch.matmul(u, block))
            block = torch.matmul(torch.t(A), T) - torch.matmul(torch.t(u), torch.matmul(torch.t(l), T))
            block, _ = torch.qr(block)
            K[:, int(i * bsize):int((i + 1) * bsize)] = block.clone().detach()
        Q, _ = torch.qr(K)
        T = torch.matmul(A, Q) - torch.matmul(l, torch.matmul(u, Q))
        Ut, St, Vt = torch.svd(T)
        S = St[0:k]
        if tpose:
            V = Ut[:, 0:k]
            U = torch.matmul(Q, Vt[:, 0:k])
        else:
            U = Ut[:, 0:k]
            V = torch.matmul(Q, Vt[:, 0:k])
        return U, S, V

    def custom_svd(self, A, k):
        U, S, V = self.random_svd(A, k)
        return U, S, torch.t(V)

    def compute_graident(self, W):
        with torch.no_grad():
            W.to('cuda')
            k = int(min(W.shape) / self._k_coeff)
            if k < 1:
                return W
            U, S, V = self.custom_svd(W, k)
            S_a = torch.ones_like(S)
            S_a[S.abs() < 1] = -1
            result = torch.matmul(U, torch.matmul(torch.diag(2 * S * S_a), V))
        return result
