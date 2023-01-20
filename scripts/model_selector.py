import torch 
from scripts.model.logic_model import  SharedLayer,SpecificLayer

def get_model(params, is_cuda=False, seed=136):
    model = {}

    torch.manual_seed(seed)
    model['enc'] = SharedLayer()
    if is_cuda: model['enc'].cuda()
    if 'and' in params['tasks']:
        model['and'] = SpecificLayer()
        if is_cuda: model['and'].cuda()
    if 'xor' in params['tasks']:
        model['xor'] = SpecificLayer()
        if is_cuda: model['xor'].cuda()
    if 'or' in params['tasks']:
        model['or'] = SpecificLayer()
        if is_cuda: model['or'].cuda()
        
    return model
