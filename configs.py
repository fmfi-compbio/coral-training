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
    "paper_bonito_f128_k7_r5": lambda: paper_eval(b_template=dict(type="paperB", repeat=5, filters=128, kernel=7, separable=True)),
    "paper_bonito_f128_k9_r5": lambda: paper_eval(b_template=dict(type="paperB", repeat=5, filters=128, kernel=9, separable=True)),
    "paper_bonito_f128_k11_r5": lambda: paper_eval(b_template=dict(type="paperB", repeat=5, filters=128, kernel=11, separable=True)),
    "paper_bonito_f128_k15_r5": lambda: paper_eval(b_template=dict(type="paperB", repeat=5, filters=128, kernel=15, separable=True)),
    "paper_bonito_f128_k21_r5": lambda: paper_eval(b_template=dict(type="paperB", repeat=5, filters=128, kernel=21, separable=True)),
    "paper_bonito_f128_k33_r5": lambda: paper_eval(b_template=dict(type="paperB", repeat=5, filters=128, kernel=33, separable=True)),
    "paper_bonito_f128_k45_r5": lambda: paper_eval(b_template=dict(type="paperB", repeat=5, filters=128, kernel=45, separable=True)),
    
    "paper_ksep_f128_k15_r5": lambda: paper_eval(b_template=dict(type="paperKSep", k=3, repeat=5, filters=128, kernel=15)),   
    "paper_ksep_f128_k21_r5": lambda: paper_eval(b_template=dict(type="paperKSep", k=3, repeat=5, filters=128, kernel=21)),
    "paper_ksep_f128_k27_r5": lambda: paper_eval(b_template=dict(type="paperKSep", k=3, repeat=5, filters=128, kernel=27)),
    "paper_ksep_f128_k33_r5": lambda: paper_eval(b_template=dict(type="paperKSep", k=3, repeat=5, filters=128, kernel=33)),
    "paper_ksep_f128_k39_r5": lambda: paper_eval(b_template=dict(type="paperKSep", k=3, repeat=5, filters=128, kernel=39)),

    "paper_s3d2_f128_k7_r5": lambda: paper_eval(b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=7, separable=True)),
    "paper_s3d2_f128_k9_r5": lambda: paper_eval(b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=9, separable=True)),
    "paper_s3d2_f128_k11_r5": lambda: paper_eval(b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=11, separable=True)),
    "paper_s3d2_f128_k13_r5": lambda: paper_eval(b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=13, separable=True)),
    "paper_s3d2_f128_k15_r5": lambda: paper_eval(b_template=dict(type="paperS2D", s2d=3, repeat=5, filters=128, s2d_filters=256, kernel=15, separable=True)),

    "paper_both_f128_k3_r5": lambda: paper_eval(b_template=dict(type="paperBoth", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=3)),
    "paper_both_f128_k9_r5": lambda: paper_eval(b_template=dict(type="paperBoth", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=9)),
    "paper_both_f128_k15_r5": lambda: paper_eval(b_template=dict(type="paperBoth", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=15)),
    "paper_both_f128_k21_r5": lambda: paper_eval(b_template=dict(type="paperBoth", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=21)),

    "paper_both_init3_f128_k3_r5": lambda: paper_eval(b_template=dict(type="paperBothInit3", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=3)),
    "paper_both_init3_f128_k9_r5": lambda: paper_eval(b_template=dict(type="paperBothInit3", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=9)),
    "paper_both_init3_f128_k15_r5": lambda: paper_eval(b_template=dict(type="paperBothInit3", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=15)),
    "paper_both_init3_f128_k21_r5": lambda: paper_eval(b_template=dict(type="paperBothInit3", s2d=3, k=3, repeat=5, filters=128, s2d_filters=256, kernel=21)),
}
