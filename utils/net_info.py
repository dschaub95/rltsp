import numpy as np
########################################
# View NN Parameters
########################################

def get_n_params1(model):
    pp = 0
    for p in list(model.parameters()):
        nn_count = 1
        for s in list(p.size()):
            nn_count = nn_count * s
        pp += nn_count
        print(nn_count)
        print(p.shape)
    print("Total: {:d}".format(pp))


def get_n_params2(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)


def get_n_params3(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_structure(model):
    print(model)