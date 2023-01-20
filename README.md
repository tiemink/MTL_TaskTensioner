# MTL_TaskTensioner

This code repository includes the source code of Task Tensioner method proposed in the [Paper](https://arxiv.org/pdf/2204.06698.pdf):

```
Leveraging convergence behavior to balance conflicting tasks in multi-task learning,
Angelica Tiemi Mizuno Nakamura, Valdir Grassi, and Denis Fernando Wolf
Neurocomputing, Volume 511, 2022.
Pages 43-53, ISSN 0925-2312.
```

# Requirements

* pytorch==1.8.1
* tqdm==4.64.1
* protobuf==3.20.0
* numpy==1.19

# Usage 

This repository includes the implementation of Task Tensioners to change the central direction between task gradients. To exemplify how to use tensioners, we also include the implementation of a multi-task learning in the problem of logical operators. Some parameters can be configured in the 'logic.json' file.

To train a model, use the command:

```
python train.py [-h] [--param_file [PARAM_FILE]] 

--param_file                JSON parameters file
```


# Contact
For any question, you can contact tiemi[dot]mizuno[at]usp[dot]br


# Citation
If you find this code useful in your research, please cite:

```
@article{NAKAMURA2022,
title = {Leveraging convergence behavior to balance conflicting tasks in multi-task learning},
author = {Angelica Tiemi Mizuno Nakamura and Valdir Grassi and Denis Fernando Wolf},
journal = {Neurocomputing},
volume = {511},
pages = {43-53},
year = {2022},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2022.09.042},
url = {https://www.sciencedirect.com/science/article/pii/S0925231222011213}
}
```