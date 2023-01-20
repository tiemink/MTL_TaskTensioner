import torch 
import json, sys
import scripts.logic_loader as data_loader
import scripts.model_selector as model_selector
from torch.autograd import Variable
from scripts.descent_dir.directions import Directions
from scripts.descent_dir.history_directions import GradHistory
import numpy as np 
from tqdm import tqdm
import argparse
from time import time as tm

def parse_args(argv):
    """
    Build parser using [<args>] form.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    #region Mandatory arguments
    ################################################################
    required = parser.add_argument_group('Required arguments')
    
    required.add_argument(
        "-param", "--param_file",
        type=str,
        required=True,
        help="JSON parameters file."
    )
    #endregion Mandatory arguments

    #region Optional arguments
    ################################################################
    optional = parser.add_argument_group('Non-required arguments')

    optional.add_argument(
        "-t", "--time",
        required=False,
        default=True,
        action="store_true",
        help="Measure execution time and print in the end."
    )
    #endregion Optional arguments

    args = parser.parse_args(args=argv)
    return args

def train_logic_gate(params_config:str, time:bool):

    start = tm()

    with open(params_config) as config_params:
        params = json.load(config_params)

    print('Loading params...')
    # set params
    min_err = params['min_err']
    tasks = params['tasks']
    is_cuda = torch.cuda.is_available()
    criterion = torch.nn.MSELoss()

    if params["use_tensors"]: directions = GradHistory(tasks,alpha=params['alpha'])
    else: directions = Directions(tasks)

    # load dataset 
    X, Y = data_loader.get_logic_data(tasks,params['loss_scale'])
    data_seq = np.arange(X.size(0))

    grads = {}
    optimizers = {}
    convergence_epoch = {t: 0.0 for t in tasks}

    # get model 
    models = model_selector.get_model(params, is_cuda)
    for m in models:
        if 'Adam' in params['optimizer']:
            optimizers[m] = torch.optim.Adam(models[m].parameters(), lr=params['lr'])
        elif 'SGD' in params['optimizer']:
            optimizers[m] = torch.optim.SGD(models[m].parameters(), lr=params['lr'], momentum=params['momentum'])

    print('Start training...')
    for epoch in tqdm(range(params['epochs'])):
        count = 0
        for m in models:
            models[m].train()

        ind = 0
        np.random.shuffle(data_seq)
        tot_loss = 0.0
        task_loss= {t: 0 for t in tasks}

        for j in range(2):

            data_point = data_seq[ind:ind+2]
            ind += 2
            x_var = Variable(X[data_point], requires_grad=False)
            if is_cuda: x_var = x_var.cuda()
            labels = {}
            for t in tasks:
                labels[t] = Variable(Y[t][data_point], requires_grad=False)
                if is_cuda: labels[t] = labels[t].cuda()

            for m in models:
                models[m].zero_grad()
                if m == 'enc':
                    continue
                optimizers[m].zero_grad()  

            Z   = models['enc'](x_var)
                
            # Task-specific outputs
            loss_tasks = {}
            for t in tasks:
                out_t = models[t](Z)

                # Task-specific losses
                loss_t = criterion(out_t, labels[t])
                
                grads[t] = torch.autograd.grad(loss_t, models['enc'].parameters(), retain_graph=True)

                loss_t.backward(retain_graph=True)
                task_loss[t] += loss_t.item()
                loss_tasks[t] = loss_t.item()
                tot_loss += loss_t.item()
                optimizers[t].step()
                optimizers[t].zero_grad()
                for parameter in models['enc'].parameters():
                    if parameter.grad is not None:
                        parameter.grad.data.zero_()

                count+=len(data_point)
            
            # compute common descent direction
            common_dir = directions.descent_direction(grads, loss=loss_tasks)
            
            # update params 
            for i_par, parameter in enumerate(models['enc'].parameters()):
                parameter.grad = common_dir[i_par].data
            optimizers['enc'].step()
        
        print(tot_loss/count, end='\r')
        for t in tasks:
            if task_loss[t]/count < min_err and convergence_epoch[t]==0.0:
                convergence_epoch[t] = epoch

        if not 0 in convergence_epoch.values():
            break

    loss_val = {}
    prediction = {} 
    for t in tasks:
        loss_val[t] = 0.0
        prediction[t] = []

    for i in range(len(X)):
        for t in tasks:
            x_var = Variable(X[i], requires_grad=False)
            if is_cuda: x_var = x_var.cuda()
            rep = models['enc'](x_var)
            pred = models[t](rep).item()
            prediction[t].append("%.3f" %pred)
            
            loss_val[t] += (Y[t][i]-pred)**2

    for t in tasks:
        loss_val[t] = np.sqrt(loss_val[t]/4).item()
        print(f"Prediction {t}: {prediction[t]}")

    print(f"Loss: {loss_val}")
    print('Runned {} epochs'.format(epoch))

    if time:
        print(f"Execution time: {(tm() - start):.2f} seconds.")

if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    param_file = args.param_file
    time = args.time

    print(f"\n{'***'*5} Running script {'***'*5}\n")
    train_logic_gate(param_file, time)
