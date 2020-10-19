import math

def variable(epochs, lr_f):
    return [(1, lr_f(i / epochs)) for i in range(epochs)]

schedules = {
    # for testing if edgetpu compiler likes the architecture
    "one": [(1, 0.01)],
    "test": [
        (6, 0.01),
    ],
    # fast schedule, decent results in 3 hours
    "fast": [
        (60, 0.01),
        (60, 0.001),
        (60, 0.0001),
    ],
    # full day schedule
    "full_orig": [
        (200, 0.01),
        (200, 0.005),
        (200, 0.002),
        (200, 0.001),
        (200, 0.0005),
        (200, 0.0002),
        (200, 0.0001),    
    ],

    "sc": variable(10, lambda x: 0.01 * (1 + 4*x)) + variable(10, lambda x: 0.05 * (1-x)),

    "linear_300_001": variable(300, lambda x: 0.01 * (1-x)),
    "linear_300_002": variable(300, lambda x: 0.02 * (1-x)),
    "linear_300_0005": variable(300, lambda x: 0.005 * (1-x)),

    "linear_500_001": variable(500, lambda x: 0.01 * (1-x)),
    "linear_500_002": variable(500, lambda x: 0.02 * (1-x)),
    "linear_500_0005": variable(500, lambda x: 0.005 * (1-x)),


    "linear_750_002": variable(750, lambda x: 0.02 * (1-x)),
    "linear_750_001": variable(750, lambda x: 0.01 * (1-x)),
    "linear_750_0005": variable(750, lambda x: 0.005 * (1-x)),
    "linear_750_0002": variable(750, lambda x: 0.002 * (1-x)),
    
    "linear_900_001": variable(900, lambda x: 0.01 * (1-x)),
    "linear_1200_001": variable(1200, lambda x: 0.01 * (1-x)),
    "linear_1500_001": variable(1500, lambda x: 0.01 * (1-x)),
    "linear_2000_001": variable(2000, lambda x: 0.01 * (1-x)),
    "linear_3000_001": variable(3000, lambda x: 0.01 * (1-x)),
    "linear_4000_001": variable(4000, lambda x: 0.01 * (1-x)),
    "linear_6000_001": variable(10, lambda x: 0.01 * x) + variable(5990, lambda x: 0.01 * (1-x)),
    "linear_8000_001": variable(25, lambda x: 0.01 * x) + variable(7975, lambda x: 0.01 * (1-x)),
    "linear_10000_001": variable(50, lambda x: 0.01 * x) + variable(9950, lambda x: 0.01 * (1-x)),


    "linear_2000_0001": variable(2000, lambda x: 0.001 * (1-x)),

    "linear_2000_0005": variable(2000, lambda x: 0.005 * (1-x)),
    "linear_2000_005": variable(2000, lambda x: 0.05 * (1-x)),
    "linear_2000_002": variable(2000, lambda x: 0.02 * (1-x)),


    "cos_750_002": variable(10, lambda x: 0.02 * x) + variable(740, lambda x: 0.02 * (1 + math.cos(x * math.pi)) / 2),
    "cos_750_001": variable(10, lambda x: 0.01 * x) + variable(740, lambda x: 0.01 * (1 + math.cos(x * math.pi)) / 2),
    "cos_750_0005": variable(10, lambda x: 0.005 * x) + variable(740, lambda x: 0.005 * (1 + math.cos(x * math.pi)) / 2),



    "linear_300_0001": variable(300, lambda x: 0.001 * (1-x)),
    "linear_500_0001": variable(500, lambda x: 0.001 * (1-x)),
    "linear_1000_0001": variable(500, lambda x: 0.001 * (1-x)),

    "exp_300": variable(300, lambda x: 0.01 * (0.01 ** x)),
    "exp_500": variable(500, lambda x: 0.01 * (0.01 ** x)),

    "cos_300_001": variable(10, lambda x: 0.01 * x) + variable(290, lambda x: 0.01 * 0.5 * (1 + math.cos(x * math.pi))),
    "cos_500_001": variable(10, lambda x: 0.01 * x) + variable(490, lambda x: 0.01 * 0.5 * (1 + math.cos(x * math.pi))),

    "const_300_0001": [(300, 0.001)],
    "const_300_001": [(300, 0.01)],

    "const_500_0001": [(500, 0.001)],
    "const_500_001": [(500, 0.001)],


    "full": [
        (100, 0.01),
        (100, 0.005),
        (100, 0.002),
        (200, 0.001),
        (200, 0.0005),
        (200, 0.0002),
        (200, 0.0001),
        (100, 0.00005),    
        (100, 0.00002),    
        (100, 0.00001),    
    ],
}