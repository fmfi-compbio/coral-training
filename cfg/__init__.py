from . import tinyB as tinyB

def dpd_config(*, activation="relu6", width=128, kernel=11, residual=True):
    common = dict(activation=activation)
    cfg = [
        #C1
        dict(**common, repeat = 1, filters = width, kernel = 9, stride = 3, residual = False, separable = False,),

        #B1
        dict(**common, type="dpd", repeat = 3, filters = width, kernel = kernel, residual = residual, separable = True,),
        #B2
        dict(**common, type="dpd", repeat = 3, filters = width, kernel = kernel, residual = residual, separable = True,),
        #B3
        dict(**common, type="dpd", repeat = 3, filters = width, kernel = kernel, residual = residual, separable = True,),
        #B4
        dict(**common, type="dpd", repeat = 3, filters = width, kernel = kernel, residual = residual, separable = True,),
        #B5
        dict(**common, type="dpd", repeat = 3, filters = width, kernel = kernel, residual = residual, separable = True,),

        #C2
        dict(**common, repeat = 1, filters = 160, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 11, residual = False, separable = False,),
        dict(type="decoder")
    ]
    return cfg

def best16():
    common = dict(activation="relu6")
    cfg = [
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 32, kernel = 9, stride = 1, residual = False, separable = False,),
        #C1
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 128, kernel = 9, stride = 3, residual = False, separable = False,),

        #B1
        dict(**common, type="dpd", repeat = 3, filters = 116, kernel = 11, residual = True, separable = True,),
        #B2
        dict(**common, type="dpd", repeat = 3, filters = 116, kernel = 11, residual = True, separable = True,),
        #B3
        dict(**common, type="dpd", repeat = 3, filters = 116, kernel = 11, residual = True, separable = True,),
        #B4
        dict(**common, type="dpd", repeat = 3, filters = 116, kernel = 11, residual = True, separable = True,),
        #B5
        dict(**common, type="dpd", repeat = 3, filters = 116, kernel = 11, residual = True, separable = True,),

        #C2
        dict(**common, repeat = 1, filters = 160, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 9, residual = False, separable = False,),
        dict(type="decoder")
    ]
    return cfg

def sloth():
    common = dict(activation="relu6")
    cfg = [
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 32, kernel = 9, stride = 1, residual = False, separable = False,),
        #C1
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 128, kernel = 9, stride = 3, residual = False, separable = False,),

        #B
        *[
            dict(**common, repeat = 4, filters = 152, kernel = 11, residual=True, separable = False,)
            for _ in range(5)
        ],

        #C2
        dict(**common, repeat = 1, filters = 160, kernel = 11, residual = False, separable = False,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 9, residual = False, separable = False,),
        dict(type="decoder")
    ]
    return cfg


def cnt_search(*, cnt, repeat):
    common = dict(activation="relu6", bn_momentum=0.9)
    cfg = [
        #C1
        dict(**common, repeat = 1, filters = 64, kernel = 9, stride = 3, residual = False, separable = False,),

        #B
        *[
            dict(**common, repeat = repeat, filters = 128, kernel = 11, residual = True, separable = True,)
            for _ in range(cnt)
        ],

        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 11, residual = False, separable = False,),
        dict(type="decoder")
    ]
    return cfg


def nores(*, width):
    common = dict(activation="relu6", dropout=0.05, bn_momentum=0.9)
    cfg = [
        #C1
        dict(**common, repeat = 1, filters = 64, kernel = 9, stride = 3, residual = False, separable = False,),

        #B
        *[
            dict(**common, repeat = 3, filters = width, kernel = 11, residual = False, separable = True,)
            for _ in range(5)
        ],

        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 11, residual = False, separable = False,),
        dict(type="decoder")
    ]
    return cfg




def ksize(*, kernel):
    common = dict(activation="relu6", dropout=0.0, bn_momentum=0.9)
    cfg = [
        #C1
        dict(**common, repeat = 1, filters = 64, kernel = 9, stride = 3, residual = False, separable = False,),

        #B1
        dict(**common, repeat = 3, filters = 128, kernel = kernel, residual = True, separable = True,),
        #B2
        dict(**common, repeat = 3, filters = 128, kernel = kernel, residual = True, separable = True,),
        #B3
        dict(**common, repeat = 3, filters = 128, kernel = kernel, residual = True, separable = True,),
        #B4
        dict(**common, repeat = 3, filters = 128, kernel = kernel, residual = True, separable = True,),
        #B5
        dict(**common, repeat = 3, filters = 128, kernel = kernel, residual = True, separable = True,),

        #C2
        dict(**common, repeat = 1, filters = 128, kernel = kernel, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 11, residual = False, separable = False,),
        dict(type="decoder")
    ]
    return cfg


def _tail(tail):
    common = dict(activation="relu6", dropout=0.05, bn_momentum=0.9)
    cfg = [
        #C1
        dict(**common, repeat = 1, filters = 64, kernel = 9, stride = 3, residual = False, separable = False,),

        #B
        *[
            dict(**common, repeat = 3, filters = 128, kernel = 11, residual=True, separable = True,)
            for _ in range(5)
        ],
        *tail,
        dict(type="decoder")
    ]
    return cfg

def tailsep():
    common = dict(activation="relu6", dropout=0.05, bn_momentum=0.9)
    return _tail([
        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 11, residual = False, separable = True,),
    ])

def tailsep2():
    common = dict(activation="relu6", bn_momentum=0.9)
    return _tail([
        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(activation=None, bn_momentum=0.9, repeat = 1, filters = 64, kernel = 11, residual = False, separable = True,),
        dict(**common, repeat = 1, filters = 64, kernel = 11, residual = False, separable = True,),
    ])

def tailx2():
    common = dict(activation="relu6", bn_momentum=0.9)
    return _tail([
        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 7, residual = False, separable = False,),
        dict(**common, repeat = 1, filters = 64, kernel = 7, residual = False, separable = False,),
    ])

def tailk(*, kernel):
    common = dict(activation="relu6", dropout=0.05, bn_momentum=0.9)
    return _tail([
        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = kernel, residual = False, separable = False,),
    ])

def tail0():
    common = dict(activation="relu6", bn_momentum=0.9)
    return _tail([
        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
    ])

def tail00():
    return _tail([])

def head(head):
    common = dict(activation="relu6", dropout=0.05, bn_momentum=0.9)
    cfg = [
        *head,
        #B
        *[
            dict(**common, repeat = 3, filters = 128, kernel = 11, residual=True, separable = True,)
            for _ in range(5)
        ],

        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 11, residual = False, separable = False,),
        dict(type="decoder")
    ]
    return cfg

def pool(*, type="pool", filters, pool_filters, pool=3, repeat=3,blocks=5, kernel=11, **kwargs):
    common = dict(activation="relu6", bn_momentum=0.9)
    cfg = [
        #C1
        dict(**common, repeat = 1, filters = 64, kernel = 9, stride = 3, residual = False, separable = False,),

        #B
        *[
            dict(**common, type=type, pool=pool, repeat = repeat, pool_filters=pool_filters, filters = filters, kernel = kernel, separable = True, **kwargs)
            for _ in range(blocks)
        ],
        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 7, residual = False, separable = False,),

        dict(type="decoder")
    ]
    return cfg



def poolhead(*, type="pool", filters, pool_filters, pool=3, repeat=3,blocks=5, kernel=11, **kwargs):
    common = dict(activation="relu6", bn_momentum=0.9)
    cfg = [
        #C1
        dict(**common, repeat = 1, filters = 16, kernel = 9, stride = 1, residual = False, separable = False,),
        dict(**common, repeat = 1, filters = 64, kernel = 9, stride = 3, residual = False, separable = False,),

        #B
        *[
            dict(**common, type=type, pool=pool, repeat = repeat, pool_filters=pool_filters, filters = filters, kernel = kernel, residual=True, separable = True, **kwargs)
            for _ in range(blocks)
        ],

        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 7, residual = False, separable = False,),
        dict(type="decoder")
    ]
    return cfg

def nosep(*, width):
    common = dict(activation="relu6", bn_momentum=0.9)
    cfg = [
        #C1
        dict(**common, repeat = 1, filters = 64, kernel = 9, stride = 3, residual = False, separable = False,),

        #B
        *[
            dict(**common, repeat = 3, filters = width, kernel = 11, residual=True, separable = False,)
            for _ in range(5)
        ],

        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 7, residual = False, separable = False,),
        dict(type="decoder")
    ]
    return cfg



def simpleres(*, type="simpleres", filters, repeat=4, blocks=5, kernel=11, **kwargs):
    common = dict(activation="relu6", bn_momentum=0.9)
    cfg = [
        #C1
        dict(**common, repeat = 1, filters = filters, kernel = 9, stride = 3, residual = False, separable = False,),

        #B
        *[
            dict(**common, type=type, repeat = repeat, filters = filters, kernel = kernel, residual=True, separable = True, **kwargs)
            for _ in range(blocks)
        ],

        #C2
        dict(**common, repeat = 1, filters = 128, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 7, residual = False, separable = False,),
        dict(type="decoder")
    ]
    return cfg

configs = {
    "default": lambda: tinyB.get_config(activation="relu6", bn_momentum=0.90, filters=128),
    #"default": lambda: tinyB.get_config(activation="relu6", bn_momentum=0.90, filters=152),

    "bneck": lambda: bneck_config(activation="relu6", width=160, kernel=11),
    "bneck15": lambda: bneck_config(activation="relu6", width=160, kernel=15),

    # tinyB
    "cnt_5_3": lambda: cnt_search(cnt=5, repeat=3),
    "cnt_6_2": lambda: cnt_search(cnt=6, repeat=2),
    "cnt_4_4": lambda: cnt_search(cnt=4, repeat=4),

    "cnt_5_4": lambda: cnt_search(cnt=5, repeat=4),

    "default148": lambda: tinyB.get_config(activation="relu6", bn_momentum=0.90, filters=148),
    "default140": lambda: tinyB.get_config(activation="relu6", bn_momentum=0.90, filters=140),
    "default132": lambda: tinyB.get_config(activation="relu6", bn_momentum=0.90, filters=132),
    "default128": lambda: tinyB.get_config(activation="relu6", bn_momentum=0.90, filters=128),
    "default120": lambda: tinyB.get_config(activation="relu6", bn_momentum=0.90, filters=120),
    "default112": lambda: tinyB.get_config(activation="relu6", bn_momentum=0.90, filters=112),


    "nosep80": lambda: nosep(width=80),
    "nosep96": lambda: nosep(width=96),
    "nosep112": lambda: nosep(width=112),
    "nosep128": lambda: nosep(width=128),
    "nosep132": lambda: nosep(width=132),
    "nosep140": lambda: nosep(width=140),
    "nosep150": lambda: nosep(width=150),
    "nosep152": lambda: nosep(width=152),

    "nores_128": lambda: nores(width=128),

    "ksize5": lambda: ksize(kernel=5),

    "ksize7": lambda: ksize(kernel=7), #
    "ksize9": lambda: ksize(kernel=9), #

    "ksize10": lambda: ksize(kernel=10), #

    "ksize11": lambda: ksize(kernel=11), #
    "ksize13": lambda: ksize(kernel=13), #

    "ksize15": lambda: ksize(kernel=15),
    "ksize19": lambda: ksize(kernel=19),
    "ksize31": lambda: ksize(kernel=31),
    "ksize47": lambda: ksize(kernel=47),

    "tailsep": lambda: tailsep(),
    "tailsep2": lambda: tailsep2(),
    "tailx2": lambda: tailx2(),

    "tailk1": lambda: tailk(kernel=1),
    "tailk3": lambda: tailk(kernel=3),
    "tailk5": lambda: tailk(kernel=5),
    "tailk7": lambda: tailk(kernel=7),
    "tailk9": lambda: tailk(kernel=9),
    "tailk11": lambda: tailk(kernel=11), # baseline
    "tailk13": lambda: tailk(kernel=13),
    "tail0": lambda: tail0(),
    "tail00": lambda: tail00(),

    "headsep": lambda: head([
        #C1
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 64, kernel = 9, stride = 3, residual = False, separable = True,),
    ]),
    "headk11": lambda: head([
        #C1
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 64, kernel = 11, stride = 3, residual = False, separable = False,),
    ]),
    "headk15": lambda: head([
        #C1
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 64, kernel = 15, stride = 3, residual = False, separable = False,),
    ]),
    "headf128": lambda: head([
        #C1
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 128, kernel = 9, stride = 3, residual = False, separable = False,),
    ]),
    "headpre": lambda: head([
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 32, kernel = 9, stride = 1, residual = False, separable = False,),
        #C1
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 128, kernel = 9, stride = 3, residual = False, separable = False,),
    ]),
    "headpre2": lambda: head([
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 32, kernel = 15, stride = 1, residual = False, separable = False,),
        #C128
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 64, kernel = 9, stride = 3, residual = False, separable = False,),
    ]),
    "headf128k33": lambda: head([
        #C1
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 128, kernel = 33, stride = 3, residual = False, separable = False,),
    ]),
    "headk7": lambda: head([
        #C1
        dict(activation="relu6",bn_momentum=0.9, repeat = 1, filters = 64, kernel = 7, stride = 3, residual = False, separable = False,),
    ]),
    "pool": lambda: pool(pool=2, filters=128, pool_filters=256),
    "pool3f140": lambda: pool(filters=140, pool_filters=280),
    "pool3f148": lambda: pool(filters=148, pool_filters=296),
    "pool3f152": lambda: pool(filters=152, pool_filters=304),
    "pool4": lambda: pool(pool=4, filters=128, pool_filters=256),
    "pool4f152x360": lambda: pool(pool=4, filters=152, pool_filters=360),

    "pool3": lambda: pool(pool=3, filters=128, pool_filters=256),
    "pool3r4": lambda: pool(pool=3, filters=128, pool_filters=256, repeat=4),
    "pool3r5": lambda: pool(pool=3, filters=128, pool_filters=256, repeat=5),

    "poolc2x128x196": lambda: pool(type="poolconv", pool=2, filters=128, pool_filters=196),

    "poolx3": lambda: pool(type="poolx"),
    "poolx3r4": lambda: pool(type="poolx", filters=128, pool_filters=256, repeat=4),
    "poolx3r4b6": lambda: pool(type="poolx", filters=128, pool_filters=256, repeat=4, blocks=6),

    "poolx4r5": lambda: pool(type="poolx", pool=4, filters=128, pool_filters=256, repeat=4),
    "poolx4r5b6": lambda: pool(type="poolx", pool=4, filters=128, pool_filters=256, repeat=4, blocks=6),

    "poolx3r5b6f128x196": lambda: pool(type="poolx", filters=128, pool_filters=196, repeat=5, blocks=6),

    "poolx3x128x384": lambda: pool(type="poolx", pool=3, filters=128, pool_filters=384),

    "poolx3r4f140x256": lambda: pool(type="poolx", pool=3, filters=140, pool_filters=256, repeat=4),

    "poolx3r4f156x216": lambda: pool(type="poolx", pool=3, filters=156, pool_filters=216, repeat=4),


    "poolx3r5": lambda: pool(type="poolx", pool=3, filters=128, pool_filters=256, repeat=5),

    "poolx3r5f128x224k7x11b6": lambda: pool(type="poolx", pool=3, filters=128, pool_filters=224, repeat=5, kernel=11, pool_kernel=7, blocks=6),


    "poolx3r5f144x196k7x11b6": lambda: pool(type="poolx", pool=3, filters=128, pool_filters=224, repeat=5, kernel=11, pool_kernel=7, blocks=6),


    "poolx3r5k7x15": lambda: pool(type="poolx", pool=3, filters=128, pool_filters=256, repeat=5, kernel=15, pool_kernel=7),



    "poolxx3r5": lambda: pool(type="poolxx", pool=3, filters=128, pool_filters=256, repeat=5),


    "pooly2r4": lambda: pool(type="pooly", pool=2, filters=128, pool_filters=256, repeat=4),

    "pooly3r4": lambda: pool(type="pooly", filters=128, pool_filters=256, repeat=4),
    "pooly3r5": lambda: pool(type="pooly", filters=128, pool_filters=256, repeat=5),


    "poolz3r4": lambda: pool(type="poolz", filters=128, pool_filters=256, repeat=4),


    "pooly3r4f140x288": lambda: pool(type="pooly", filters=104, pool_filters=288, repeat=4),
    "pooly3r4f156x196": lambda: pool(type="pooly", filters=156, pool_filters=196, repeat=4),

    "pooly3r4b6": lambda: pool(type="pooly", filters=128, pool_filters=256, repeat=4, blocks=6),
    "poolc2": lambda: pool(type="poolc", pool=2, filters=128, pool_filters=256, repeat=3),
    "poolc3": lambda: pool(type="poolc", pool=3, filters=128, pool_filters=256, repeat=3),

    "poolf3r4f144": lambda: pool(type="poolf", filters=144, pool_filters=256, repeat=4),
    "poolf3r4f144x296": lambda: pool(type="poolf", filters=144, pool_filters=296, repeat=4),

    "poolf3r4f144x224": lambda: pool(type="poolf", filters=144, pool_filters=224, repeat=4),

    "poolf3r5f144": lambda: pool(type="poolf", filters=144, pool_filters=256, repeat=5),


    "poolff3r4f148x280": lambda: pool(type="poolff", filters=148, pool_filters=280, repeat=4),
    "poolff3r5f136x272": lambda: pool(type="poolff", filters=136, pool_filters=272, repeat=5),

    "poolsf3r5f136x272": lambda: pool(type="poolsf", filters=136, pool_filters=272, repeat=5),

    "poolg3r4": lambda: pool(type="poolg", filters=128, pool_filters=256, repeat=4),
    "poolh3r4": lambda: pool(type="poolh", filters=128, pool_filters=256, repeat=4),


    "dpd40": lambda: dpd_config(width=40),
    "dpd48": lambda: dpd_config(width=48),
    "dpd56": lambda: dpd_config(width=56),
    "dpd60": lambda: dpd_config(width=60),

    "dpd64": lambda: dpd_config(width=64),
    "dpd72": lambda: dpd_config(width=72),
    "dpd80": lambda: dpd_config(width=80),

    "dpd96": lambda: dpd_config(width=96),
    "dpd116": lambda: dpd_config(width=116),
    "dpdplain": lambda: dpd_config(width=150, residual=False),
    "best16": lambda: best16(),
    "dilation": lambda: tinyB.get_config(activation="relu6", bn_momentum=0.90, dilation=2),

    "sloth": lambda: sloth(),

    "simpleresr4b5f128": lambda: simpleres(filters=128),
    "simpleresr1b20f128": lambda: simpleres(filters=128,repeat=1, blocks=20),
    "simpledpdr1b20f144": lambda: simpleres(type="simpledpd", filters=144,repeat=1, blocks=20),
    "simpledpdr4b5f144": lambda: simpleres(type="simpledpd", filters=144,repeat=4, blocks=5),
    "simpleexpr1b20f128x160": lambda: simpleres(type="simpleexp", filters=128,exp_filters=160, repeat=1, blocks=20),

    "poolxhead3r5": lambda: poolhead(type="poolx", pool=3, filters=128, pool_filters=256, repeat=5),
    "simpleeypr1b20f128x160": lambda: simpleres(type="simpleeyp", filters=128,exp_filters=160, repeat=1, blocks=20),


    "poolx3r5f144x256k9x9": lambda: pool(type="poolx", pool=3, filters=144, pool_filters=256, kernel=9, pool_kernel=9, repeat=5),
    "poolx3r5f156x224k9x9": lambda: pool(type="poolx", pool=3, filters=156, pool_filters=224, kernel=9, pool_kernel=9, repeat=5),
    "poolxu3r5": lambda: pool(type="poolxu", pool=3, filters=128, pool_filters=256, repeat=5),

    "poolx3r4b6": lambda: pool(type="poolx", pool=3, filters=128, pool_filters=256, repeat=4,blocks=6),
    "poolx3r5b6": lambda: pool(type="poolx", pool=3, filters=128, pool_filters=256, repeat=5,blocks=6),
    "poolx3r6": lambda: pool(type="poolx", pool=3, filters=128, pool_filters=256, repeat=6,blocks=5),

    "poolffd13r5f148x280": lambda: pool(type="poolff", depth_multiplier=1, filters=148, pool_filters=280, repeat=5),
    "pooljj3r5f148x288": lambda: pool(type="pooljj", filters=148, pool_filters=288, repeat=5),

    "poolxv3r5": lambda: pool(type="poolxv", pool=3, filters=128, pool_filters=256, repeat=5),
    "poolx3r5f144x288": lambda: pool(type="poolx", pool=3, filters=144, pool_filters=288, repeat=5),

    "poolx3r5initi": lambda: pool(type="poolxiniti", pool=3, filters=128, pool_filters=256, repeat=5),
    "poolx3r5initii": lambda: pool(type="poolxinitii", pool=3, filters=128, pool_filters=256, repeat=5),
    "poolx3r5initj": lambda: pool(type="poolxinitj", pool=3, filters=128, pool_filters=256, repeat=5),

    "poolx3r5bn": lambda: pool(type="poolxbn", pool=3, filters=128, pool_filters=256, repeat=5),

    "poolerasea3r5": lambda: pool(type="poolerasea", pool=3, filters=128, pool_filters=256, repeat=5),

    # TODO


    "poolxy3r5": lambda: pool(type="poolxy", pool=3, filters=128, pool_filters=256, repeat=5),

    "simpleexpr4b5f128x160": lambda: simpleres(type="simpleexp", filters=128,exp_filters=160, repeat=4, blocks=5),
    "simpleeypr4b5f128x160": lambda: simpleres(type="simpleeyp", filters=128,exp_filters=160, repeat=4, blocks=5),
    "simpleezpr1b20f128x160": lambda: simpleres(type="simpleezp", filters=128,exp_filters=160, repeat=1, blocks=20),
    "simpleezpr4b5f128x160": lambda: simpleres(type="simpleezp", filters=128,exp_filters=160, repeat=4, blocks=5),

}
