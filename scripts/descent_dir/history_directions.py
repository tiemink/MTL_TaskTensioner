import torch, copy
import numpy as np 
from collections import defaultdict
from scripts.descent_dir.directions import Directions
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class GradHistory(Directions):
    MAX_BATCH_ITER = 10 
    def __init__(self, tasks, alpha):
        super().__init__(tasks)
        self._init_grad_history()
        self.alpha=alpha

    def _init_grad_history(self):
        """ Initialize variables"""

        self.grad_accumulation = defaultdict(list)
        self.squared_grad_accumulation = defaultdict(list)
        self.previous_grad = defaultdict(list)
        self.iterations = 0
        self.grad_mean_prev = []

    def _accumulate_gradient_vector_slide(self, grad_vector:dict)->None:

        if self.iterations == 0:
            self.grad_accumulation = {t: torch.zeros(self.MAX_BATCH_ITER) for t in self.tasks}
        
        idx = self.iterations%self.MAX_BATCH_ITER
        for t in self.tasks:
            self.grad_accumulation[t][idx] = torch.norm(parameters_to_vector(grad_vector[t]))

    def _compute_tension(self,parametrization):
        a=self.alpha
        tension = (a/(1+np.exp(-parametrization*np.exp(1)+np.exp(1)))+1-a)
        return max(0, tension)


    def _angle_between_vectors(self, vector1, vector2):

        # input: vector parameters 
        rad = torch.arccos(torch.dot(vector1, vector2)/(torch.norm(vector1)*torch.norm(vector2))).item()
        return rad

    def descent_direction(self, grad_params: dict, loss=None)->list:
       
        # Compute common direction between tasks (center direction)
        new_dir = self._descent_direction(grad_params, self.tasks)

        self._accumulate_gradient_vector_slide(grad_params)
        self.iterations += 1  

        if self.iterations>=self.MAX_BATCH_ITER:
            # compute mean of gradients
            grad_accumulation_mean = {t: torch.sum(self.grad_accumulation[t]) for t in self.tasks}

            if self.grad_mean_prev:
                tmp_grad = copy.deepcopy(grad_params[self.tasks[0]])
                bisec_vector = parameters_to_vector(new_dir)
                tension_vector = []
                vector_param = defaultdict(list)
                for t in self.tasks:

                    # Compute relative change
                    vector_param[t] = parameters_to_vector(grad_params[t])
                    grad_parametrization = grad_accumulation_mean[t]/self.grad_mean_prev[t] + np.log10(loss[t])
                    
                    # Compute tension factor
                    tension_task = self._compute_tension(grad_parametrization.item())

                    # Compute tensions of each task
                    diff_vec = vector_param[t]-bisec_vector
                    unit_vector = diff_vec/torch.norm(diff_vec)

                    if tension_vector == []:
                        tension_vector = unit_vector*tension_task
                    else:
                        tension_vector += unit_vector*tension_task

                # Get new descent direction
                tension_vector = bisec_vector + tension_vector
                
                ########################
                ###### check direction
                ##############################
                for t in self.tasks:
                    angle = self._angle_between_vectors(vector_param[t], tension_vector)*180/np.pi
                    if angle>90:
                        unit_vector_param = vector_param[t]/torch.norm(vector_param[t])
                        w = tension_vector - torch.dot(tension_vector, unit_vector_param)*unit_vector_param
                        w_unit = w/torch.norm(w)
                        alpha_angle = np.pi-self._angle_between_vectors(vector_param[t], (tension_vector-vector_param[t]))
                        tension_vector = np.tan(alpha_angle)*torch.norm(vector_param[t])*w_unit
                    
                ##########################################

                vector_to_parameters(tension_vector, tmp_grad)
                new_dir = tmp_grad
                vector_param = []
                tmp_grad = []
                tension_vector = []

            self.grad_mean_prev = grad_accumulation_mean
        
        return  new_dir


if __name__ == "__main__":
    
    tasks = ['a', 'b']
    alpha = 0.0
    directions = GradHistory(tasks, alpha)

    loss = {}
    loss['a'] = 1.2
    loss['b'] = 1.6

    grads = {}
    grads['a'] = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    grads['b'] = torch.tensor([[7.,8.,9.], [10.,11.,12.]])


    common_dir = directions.descent_direction(grads, loss)

    print(common_dir)