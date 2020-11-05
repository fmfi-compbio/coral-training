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

def poolinit(*, type="pool", filters, pool_filters, pool=3, repeat=3,blocks=5, kernel=11, **kwargs):
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
        dict(**common, type="sepinit", filters = 128, kernel = 11),
        #C3
        dict(**common, type="convinit", filters = 64, kernel = 7),
        dict(type="decoder", init='zeros')
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


def blockd(*, activation="relu6", type="blockD", width=128, kernel=5, repeat=5, blocks=5):
    common = dict(activation=activation)
    cfg = [
        #C1
        dict(**common, repeat = 1, filters = width, kernel = 9, stride = 3, residual = False, separable = False,),

        #B1
        *[
            dict(**common, type=type, repeat = repeat, filters = width, kernel = kernel)
            for _ in range(blocks)
        ],
        #C2
        dict(**common, repeat = 1, filters = 160, kernel = 11, residual = False, separable = True,),
        #C3
        dict(**common, repeat = 1, filters = 64, kernel = 11, residual = False, separable = False,),
        dict(type="decoder")
    ]
    return cfg



def paper_eval(*, b_template, filters=128, block_cnt=5):
    cfg = [
        #C1
        dict(type="paperC", filters = filters, kernel = 9, stride = 3, separable = False,),
        #B1-5
        *[b_template for _ in range(block_cnt)],
        #C2
        dict(type="paperC", filters = filters, kernel = 11, separable = True,),
        #C3
        dict(type="paperC", filters = 64, kernel = 7, separable = False,),
        #Decoder
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

    "poolx3r4b6": lambda: pool(type="poolx", pool=3, filters=128, pool_filters=256, repeat=4,blocks=6),
    "poolx3r5b6": lambda: pool(type="poolx", pool=3, filters=128, pool_filters=256, repeat=5,blocks=6),
    "poolx3r6": lambda: pool(type="poolx", pool=3, filters=128, pool_filters=256, repeat=6,blocks=5),


    "poolx3r5f144x288": lambda: pool(type="poolx", pool=3, filters=144, pool_filters=288, repeat=5),

    "poolx3r5initi": lambda: pool(type="poolxiniti", pool=3, filters=128, pool_filters=256, repeat=5),
    "poolx3r5initii": lambda: pool(type="poolxinitii", pool=3, filters=128, pool_filters=256, repeat=5),
    "poolx3r5initiii": lambda: pool(type="poolxinitiii", pool=3, filters=128, pool_filters=256, repeat=5),
    "poolx3r5initiv": lambda: pool(type="poolxinitiv", pool=3, filters=128, pool_filters=256, repeat=5),
    "poolx3r5initv": lambda: pool(type="poolxinitv", pool=3, filters=128, pool_filters=256, repeat=5),

    "poolx3r5initj": lambda: pool(type="poolxinitj", pool=3, filters=128, pool_filters=256, repeat=5),

    "poolx3r5bn": lambda: pool(type="poolxbn", pool=3, filters=128, pool_filters=256, repeat=5),

    "poolerasea3r5": lambda: pool(type="poolerasea", pool=3, filters=128, pool_filters=256, repeat=5),
    "pooleraseb3r5": lambda: pool(type="pooleraseb", pool=3, filters=128, pool_filters=256, repeat=5),
    "poolerasec3r5": lambda: pool(type="poolerasec", pool=3, filters=128, pool_filters=256, repeat=5),
    "poolerased3r5": lambda: pool(type="poolerased", pool=3, filters=128, pool_filters=256, repeat=5),
    "poolerasedx3r5": lambda: pool(type="poolerasedx", pool=3, filters=128, pool_filters=256, repeat=5),
    "poolerasedd3r5": lambda: pool(type="poolerasedd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=11, pool_kernel=5,),
    "pooleraseddd3r5": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=11, pool_kernel=5,),
    "pooleraseedd3r5": lambda: pool(type="pooleraseedd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=11, pool_kernel=5,),
    "poolerasee3r5": lambda: pool(type="poolerasee", pool=3, filters=128, pool_filters=256, repeat=5),

    # local optimization of kernel sizes
    "pooleraseddd3r5k9x5": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=9, pool_kernel=5,),
    "pooleraseddd3r5k7x5": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=7, pool_kernel=5,),
    "pooleraseddd3r5k5x5": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=5, pool_kernel=5,),
    "pooleraseddd3r5k3x5": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=3, pool_kernel=5,),
    "pooleraseddd3r5k1x5": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=1, pool_kernel=5,),

    "pooleraseddd3r5k11x7": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=9, pool_kernel=7,),
    "pooleraseddd3r5k11x3": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=9, pool_kernel=3,),


    "poolerasedddinit3r5": lambda: pool(type="poolerasedddinit", pool=3, filters=128, pool_filters=256, repeat=5, kernel=11, pool_kernel=5,),

    "poolerasedddr3r5": lambda: pool(type="poolerasedddr", pool=3, filters=128, pool_filters=256, repeat=5, kernel=11, pool_kernel=5,),
    "poolerasedddq3r5": lambda: pool(type="poolerasedddq", pool=3, filters=128, pool_filters=256, repeat=5, kernel=11, pool_kernel=5,),
    "poolerasedddnobias3r5": lambda: pool(type="poolerasedddnobias", pool=3, filters=128, pool_filters=256, repeat=5, kernel=11, pool_kernel=5,),

    "pooleraseddd3r5k0x5f112x272": lambda: pool(type="pooleraseddd", pool=3, filters=112, pool_filters=272, repeat=5, kernel=0, pool_kernel=5,),
    "pooleraseddd3r5k0x5f128x256": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=0, pool_kernel=5,),
    "pooleraseddd3r5k0x5f136x256": lambda: pool(type="pooleraseddd", pool=3, filters=136, pool_filters=256, repeat=5, kernel=0, pool_kernel=5,),

    "pooleraseddd3r5k0x5f156x180": lambda: pool(type="pooleraseddd", pool=3, filters=156, pool_filters=180, repeat=5, kernel=0, pool_kernel=5,),
    "pooleraseddd3r5k0x5f136x272": lambda: pool(type="pooleraseddd", pool=3, filters=136, pool_filters=272, repeat=5, kernel=0, pool_kernel=5,),


    "pooleraseddd3r6k0x5": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=6, kernel=0, pool_kernel=5,),
    "pooleraseddd3r5b6k0x5": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=0, pool_kernel=5, blocks=6),
    "pooleraseddd3r4b7k0x5": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=4, kernel=0, pool_kernel=5, blocks=7),
    "pooleraseddd3r4b8k0x5": lambda: pool(type="pooleraseddd", pool=3, filters=128, pool_filters=256, repeat=4, kernel=0, pool_kernel=5, blocks=8),
    "pooleraseddd3r4b6k0x5f136x272": lambda: pool(type="pooleraseddd", pool=3, filters=136, pool_filters=272, repeat=4, kernel=0, pool_kernel=5, blocks=6),

    "pooleraseddd3r5b7k0x5f112x224": lambda: pool(type="pooleraseddd", pool=3, filters=112, pool_filters=224, repeat=5, kernel=0, pool_kernel=5, blocks=7),

    "poolerasedddiniti3r5": lambda: pool(type="poolerasedddiniti", pool=3, filters=128, pool_filters=256, repeat=5, kernel=11, pool_kernel=5,),
    "poolerasedddinitii3r5": lambda: pool(type="poolerasedddinitii", pool=3, filters=128, pool_filters=256, repeat=5, kernel=11, pool_kernel=5,),
    "poolerasedddinitx3r5": lambda: pool(type="poolerasedddinitx", pool=3, filters=128, pool_filters=256, repeat=5, kernel=11, pool_kernel=5,),

    "poolerasec3r5k11x13": lambda: pool(type="poolerasec", pool=3, filters=128, pool_filters=256, repeat=5, kernel=11, pool_kernel=13),
    "poolerasecinit3r5": lambda: pool(type="poolerasecinit", pool=3, filters=128, pool_filters=256, repeat=5),

    "poolx3r5initiibn": lambda: pool(type="poolxinitiibn", pool=3, filters=128, pool_filters=256, repeat=5),

    "poolx3r5initiifull": lambda: poolinit(type="poolxinitii", pool=3, filters=128, pool_filters=256, repeat=5),

    "blockd": lambda: blockd(),
    "blockdinit": lambda: blockd(type="blockdinit"),

    # TODO
    "poolxd3r5": lambda: pool(type="poolxd", pool=3, filters=128, pool_filters=256, repeat=5, kernel=5, pool_kernel=5),


    "poolxy3r5": lambda: pool(type="poolxy", pool=3, filters=128, pool_filters=256, repeat=5),

    "simpleexpr4b5f128x160": lambda: simpleres(type="simpleexp", filters=128,exp_filters=160, repeat=4, blocks=5),
    "simpleeypr4b5f128x160": lambda: simpleres(type="simpleeyp", filters=128,exp_filters=160, repeat=4, blocks=5),
    "simpleezpr1b20f128x160": lambda: simpleres(type="simpleezp", filters=128,exp_filters=160, repeat=1, blocks=20),
    "simpleezpr4b5f128x160": lambda: simpleres(type="simpleezp", filters=128,exp_filters=160, repeat=4, blocks=5),

    "paper_bonito_f128_k7_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperB", repeat=5, filters=128, kernel=7, separable=True)),
    "paper_bonito_f128_k9_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperB", repeat=5, filters=128, kernel=9, separable=True)),
    "paper_bonito_f128_k11_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperB", repeat=5, filters=128, kernel=11, separable=True)),
    "paper_bonito_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperB", repeat=5, filters=128, kernel=15, separable=True)),
    "paper_bonito_f128_k21_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperB", repeat=5, filters=128, kernel=21, separable=True)),
    "paper_bonito_f128_k33_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperB", repeat=5, filters=128, kernel=33, separable=True)),
    "paper_bonito_f128_k45_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperB", repeat=5, filters=128, kernel=45, separable=True)),
    
    "paper_bonito_init_f128_k11_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBInit", repeat=5, filters=128, kernel=11, separable=True)),

    "paper_ksep_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperKSep", k=3, repeat=5, filters=128, kernel=15)),   
    "paper_ksep_f128_k21_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperKSep", k=3, repeat=5, filters=128, kernel=21)),
    "paper_ksep_f128_k27_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperKSep", k=3, repeat=5, filters=128, kernel=27)),
    "paper_ksep_f128_k33_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperKSep", k=3, repeat=5, filters=128, kernel=33)),
    "paper_ksep_f128_k39_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperKSep", k=3, repeat=5, filters=128, kernel=39)),

    "paper_s3d2_f128_k7_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=7, separable=True)),
    "paper_s3d2_f128_k9_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=9, separable=True)),
    "paper_s3d2_f128_k11_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=11, separable=True)),
    "paper_s3d2_f128_k13_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=13, separable=True)),
    "paper_s3d2_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=15, separable=True)),

    "paper_both_f128_k3_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBoth", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=3)),
    "paper_both_f128_k6_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBoth", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=6)),
    "paper_both_f128_k6x_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBoth", s2d=3, k=2, repeat=5, filters=128, s2d_filters=256, kernel=6)),

    "paper_both_f128_k9_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBoth", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=9)),
    "paper_both_f128_k12_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBoth", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=12)),
    "paper_both_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBoth", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=15)),
    "paper_both_f128_k21_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBoth", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=21)),

    "paper_both_init_f128_k9_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=9)),
    "paper_both_init_f128_k12_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=12)),
    "paper_both_init_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=15)),

# Experimental
    "paper_both_init2_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit2", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=15)),

    "paper_both_init3_f128_k3_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit3", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=3)),
    "paper_both_init3_f128_k6_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit3", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=6)),

    "paper_both_init3_f128_k9_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit3", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=9)),
    "paper_both_init3_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit3", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=15)),
    "paper_both_init3_f128_k21_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit3", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=21)),



    "paper_both_init4_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit4", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=15)),
    "paper_both_init5_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit5", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=15)),

    "paper_both_inita_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInitA", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=15)),
    "paper_both_initb_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInitB", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=15)),
    "paper_both_initc_f128_k15_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInitC", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=15)),


    "paper_both_init_f128_k9_r5_b6": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=9), block_cnt=6),
    "paper_both_init_f128_k9_r6_b5": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit", s2d=3, k=3, repeat=6, filters=128, s2d_filters=256, kernel=9), block_cnt=5),
    "paper_both_init_f128_k9_r6_b6": lambda: paper_eval(filters=128, b_template=dict(type="paperBothInit", s2d=3, k=3, repeat=6, filters=128, s2d_filters=256, kernel=9), block_cnt=6),

# TODO:
    "paper_s3d2_f128_k11_r5_b7": lambda: paper_eval(filters=128, b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=11, separable=True), block_cnt=7),
    "paper_s3d2_f156x288_k11_r5": lambda: paper_eval(filters=156, b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=156, s2d_filters=288, kernel=11, separable=True)),
    "paper_s3d2_init_f128_k11_r5": lambda: paper_eval(filters=128, b_template=dict(type="paperS2DInit", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=11, separable=True)),
    "special_init_f128x256_r5": lambda: paper_eval(filters=128, b_template=dict(type="special", s2d=3, repeat=5, filters=128, s2d_filters=256, special_filters=128, kernel=3)),
    "special_init_f128x256_k4_r5": lambda: paper_eval(filters=128, b_template=dict(type="special", s2d=3, repeat=5, filters=128, s2d_filters=256, special_filters=128, kernel=4)),
    "special_init_f128x256x180_r5": lambda: paper_eval(filters=128, b_template=dict(type="special", s2d=3, repeat=5, filters=128, s2d_filters=256, special_filters=180, kernel=3)),

    "special2_init_f128x256_r5": lambda: paper_eval(filters=128, b_template=dict(type="special2", s2d=3, repeat=5, filters=128, s2d_filters=256, special_filters=128, kernel=3)),
    "special2_init_f128x256x160_r5": lambda: paper_eval(filters=128, b_template=dict(type="special2", s2d=3, repeat=5, filters=128, s2d_filters=256, special_filters=160, kernel=3)),

    "special2_init_f128x256x180dw5_r5": lambda: paper_eval(b_template=dict(type="special2", s2d=3, repeat=5, filters=128, s2d_filters=256, special_filters=180, kernel=[3, 5, 3], dilation=[1,3,1], final_kernel=9)),
    "special2_init_f128x256x180dw5dil132_r5": lambda: paper_eval(b_template=dict(type="special2", s2d=3, repeat=5, filters=128, s2d_filters=256, special_filters=180, kernel=[3, 5, 3], dilation=[1,3,2], final_kernel=9)),
    "special2_init_f128x256x180dw5dil133_r5": lambda: paper_eval(b_template=dict(type="special2", s2d=3, repeat=5, filters=128, s2d_filters=256, special_filters=180, kernel=[3, 5, 3], dilation=[1,3,3], final_kernel=9)),
    "special2_init_f128x256x180dw5dil231_r5": lambda: paper_eval(b_template=dict(type="special2", s2d=3, repeat=5, filters=128, s2d_filters=256, special_filters=180, kernel=[3, 5, 3], dilation=[2,3,1], final_kernel=9)),

}
