import numpy as np 
from scripts.descent_dir.min_norm_solvers import MinNormSolver

class Directions:
    
    def __init__(self, tasks):
        self.tasks = tasks
        self.solver = MinNormSolver(tasks)
        self.iteration = 0.0

    @staticmethod
    def _compute_norm_factor(grad_vector:list, norm_type='l2', loss=None)->float:
        if norm_type == 'l2':
            return np.sqrt(np.sum([gr.pow(2).sum().item() for gr in grad_vector]))
        elif norm_type == 'loss+':
            return loss * np.sqrt(np.sum([gr.pow(2).sum() for gr in grad_vector]))
        elif norm_type == None:
            return 1.0
        else:
            print('ERROR: Invalid Normalization Type')

    
    def normalize_grad_vector(self, grad_vector:list, norm_factor:float)->list:
        # input: grad parameters 
        return [grad/norm_factor for grad in grad_vector]

    def descent_direction(self, grad_params: dict, norm_type=None, loss=None)->list:
        return self._descent_direction(grad_params, norm_type, loss)

    def _descent_direction(self, grad_params:dict, norm_type:str=None, loss:str=None)->list:

        # Normalize gradients 
        norm_factor = {t: self._compute_norm_factor(grad_params[t]) for t in self.tasks}
        normalized_vector = {t: self.normalize_grad_vector(grad_params[t], norm_factor[t]) for t in self.tasks}

        # Compute weight coefficients and gamma to find center direction
        weight_coef = self.solver.find_min_norm_element_FW(normalized_vector)
        gamma = 1/sum(weight_coef[t]/norm_factor[t] for t in self.tasks)
        
        common_direction = self.combine_gradients(normalized_vector, weight_coef)
        common_direction = [vector*gamma for vector in common_direction]
        return common_direction

    def combine_gradients(self, grads, weights=None):
        if not weights:
            weights = {t: 1.0 for t in self.tasks}

        common_direction = [0]*len(grads[self.tasks[0]])
        for t in self.tasks:
            for idx, grad in enumerate(grads[t]):
                common_direction[idx] += weights[t]*grad

        return common_direction