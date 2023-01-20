import torch 

def _logic_data(task):
    
    logic_data = {'xor': [0,1,1,0],
                  'and': [0,0,0,1],
                  'or': [0,1,1,1]}
    
    return logic_data[task]
    
def get_logic_data(tasks, loss_scale:dict={'xor':10, 'and':1, 'or':1}):
    X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
    Y = {}

    for t in tasks:
        scale = loss_scale[t] if t in loss_scale.keys() else 1.0
        Y[t] = scale*torch.Tensor(_logic_data(t)).view(-1,1)

    return X, Y

    