

def get_config(*, activation, bn_momentum, dilation=1, filters=128):
    common = dict(activation=activation, bn_momentum=bn_momentum, dilation=dilation)
    cfg = [
        #C1
        dict(**common, repeat = 1, filters = 64, kernel = 9, stride = 3, residual = False, separable = False,),

        #B1
        dict(**common, repeat = 3, filters = filters, kernel = 11, residual = True, separable = True,),
        #B2
        dict(**common, repeat = 3, filters = filters, kernel = 11, residual = True, separable = True,),
        #B3
        dict(**common, repeat = 3, filters = filters, kernel = 11, residual = True, separable = True,),
        #B4
        dict(**common, repeat = 3, filters = filters, kernel = 11, residual = True, separable = True,),
        #B5
        dict(**common, repeat = 3, filters = filters, kernel = 11, residual = True, separable = True,),

        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 11, residual = False, separable = False,),
    ]
    return cfg