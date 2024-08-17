import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv as GATConv_v1
from torch_geometric.nn import GATv2Conv as GATConv

def cox_sort(out, tempcensor, temp_event_time):
    """
    out: 模型输出的风险值
    tempsurvival: 生存时间

    """
    sort_idx = torch.argsort(temp_event_time, descending=True)
    tempcensor = tempcensor[sort_idx]
    temp_event_time = temp_event_time[sort_idx]
    risklist = out[sort_idx]

    risklist = risklist.to(out.device)
    tempcensor = tempcensor.to(out.device)
    temp_event_time = temp_event_time.to(out.device)

    return risklist, tempcensor, temp_event_time

    # updated_feature_list = []

    # risklist = out[sort_idx]
    # tempsurvival = tempsurvival[sort_idx]
    # tempphase = tempphase[sort_idx]
    # tempmeta = tempmeta[sort_idx]
    # for idx in sort_idx.cpu().detach().tolist():
    #     EpochID.append(tempID[idx])
    # tempstage = tempstage[sort_idx]

    # risklist = risklist.to(out.device)
    # tempsurvival = tempsurvival.to(out.device)
    # tempphase = tempphase.to(out.device)
    # tempmeta = tempmeta.to(out.device)

    # for riskval, survivalval, phaseval, stageval, metaval in zip(risklist, tempsurvival,
    #                                                              tempphase, tempstage,
    #                                                              tempmeta):
    #     EpochSurv.append(survivalval.cpu().detach().item())
    #     EpochPhase.append(phaseval.cpu().detach().item())
    #     EpochRisk.append(riskval.cpu().detach().item())
    #     EpochStage.append(stageval.cpu().detach().item())

    # return risklist, tempsurvival, tempphase, tempmeta, EpochSurv, EpochPhase, EpochRisk, EpochStage

class coxph_loss(torch.nn.Module):

    def __init__(self):
        super(coxph_loss, self).__init__()

    def forward(self, risk, phase, censors):

        #riskmax = risk
        riskmax = F.normalize(risk, p=2, dim=0)

        log_risk = torch.log((torch.cumsum(torch.exp(riskmax), dim=0)))
        print('log_risk')
        print(log_risk)
        print(log_risk.shape)

        uncensored_likelihood = torch.add(riskmax, -log_risk)
        print('uncensored_likelihood')
        print(uncensored_likelihood)
        print(uncensored_likelihood.shape)
        
        resize_censors = censors.resize_(uncensored_likelihood.size()[0], 1)
        print('resize_censors')
        print(resize_censors)
        print(resize_censors.shape)
        
        censored_likelihood = torch.mul(uncensored_likelihood, resize_censors)
        print('censored_likelihood')
        print(censored_likelihood)
        print(censored_likelihood.shape)

        loss = -torch.sum(censored_likelihood) / float(censors.nonzero().size(0))
        #loss = -torch.sum(censored_likelihood) / float(censors.size(0))

        return loss

def non_decay_filter(model):

    no_decay = list()
    decay = list()

    for m in model.modules():
        if isinstance(m, nn.Linear):
            decay.append(m.weight)
            if m.bias != None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            no_decay.append(m.weight)
            if m.bias != None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.LayerNorm):
            no_decay.append(m.weight)
            if m.bias != None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.PReLU):
            no_decay.append(m.weight)
        elif isinstance(m, GATConv):
            decay.append(m.att)
            if m.bias != None:
                no_decay.append(m.bias)
            no_decay.append(m.position_bias)
        elif isinstance(m, GATConv_v1):
            pass
            # decay.append(m.att_l)
            # decay.append(m.att_r)
            # if m.bias != None:
            #     no_decay.append(m.bias)
            # no_decay.append(m.position_bias)
            # no_decay.append(m.angle_bias)
            # decay.append(m.att_edge_attr_pos)
            # decay.append(m.att_edge_attr_angle)

    model_parameter_groups = [dict(params=decay), dict(params=no_decay, weight_decay=0.0)]

    return model_parameter_groups