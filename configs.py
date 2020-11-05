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
