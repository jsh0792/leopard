import torch
import math

class NLLSurvLoss(object):
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))         
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def dynamic_KL_w(base_w, task_loss, decay_rate=1):
    eps = 1e-3
    return base_w / (1 + decay_rate * (task_loss + eps))

def task_loss_w(loss, alpha=1):
    return 1 / (1 + alpha * loss)

def absolute_loss_without_c(y_true, y_pred, c):
    # Create a mask tensor where c is not present
    mask = (c != 0)  # Assuming c is a torch tensor

    # Apply the mask to select only those elements
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    # Calculate absolute loss
    loss = torch.mean(torch.abs(y_true_masked - y_pred_masked))
    
    return loss