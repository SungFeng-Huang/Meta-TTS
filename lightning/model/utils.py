
def reset_LN(module):
    module.reset_parameters()
    module.elementwise_affine = False
    for p in module.parameters():
        p.requires_grad = False

