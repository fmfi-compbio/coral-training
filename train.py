import argparse
import random
import tensorflow as tf

import cfg
import net
import converter
import data_loader
import align_stats
import schedules

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--schedule", default="full")
    parser.add_argument("--data_dir", default="/scratch/basecall_train")
    parser.add_argument("--batch_size", default=15, type=int)
    parser.add_argument("--outname")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--tf_seed", default=None, type=int)
    parser.add_argument("--get_batch_seed", default=4747, type=int)
    parser.add_argument("--smooth", default=0, type=float)
    args = parser.parse_args()

    if args.tf_seed:
        tf.random.set_seed(args.tf_seed)
    # Prepare model
    schedule = schedules.schedules[args.schedule]
    net_cfg = cfg.configs[args.config]()

    model = net.make_model(net_cfg)

    def get_optimizer(variant):
        if variant=="adam":
            return tf.keras.optimizers.Adam(beta_1=0.9)
        if variant=="novograd":
            import tensorflow_addons as tfa
            return tfa.optimizers.NovoGrad(beta_1=0.9)


    model.compile(optimizer=get_optimizer(args.optimizer), loss=net.CTCLoss(smooth=args.smooth))
    print(model.summary())

    # load files
    X, Y = data_loader.load_dir(args.data_dir)

    rng = random.Random(args.get_batch_seed)
    SEQLEN=5000
    SEQLEN=5004
    #SEQLEN=600
    #SEQLEN=3000

    def get_batch(*, batch_size=100, leng=SEQLEN):
        x, y = data_loader.prep_batch(X, Y, rng, batch_size, leng)
        res = (
            tf.constant(x),
            tf.constant(y, dtype=tf.int32)
        )
        return res

    x_test, y_test = get_batch(batch_size=100, leng=SEQLEN)
    print(x_test[0])

    epoch = 1

    def _save(model, fname):
        quant = converter.convert(model, get_batch, STEPS=SEQLEN, BATCH_SIZE=4)
        with open(fname, "wb") as f:
            f.write(quant)
        print(fname, len(quant)) 
        

    #f_stat = open(f"models/{args.outname or args.config}.stat", "w")
    print("Starting training loop")
    for (n_epochs, lr) in schedule:
        tf.keras.backend.set_value(model.optimizer.lr, lr)
        for _ in range(n_epochs):
            B_SIZE = args.batch_size
            B_CNT = 1500 // B_SIZE

            x, y = get_batch(batch_size=1500, leng=SEQLEN)
            model.fit(x, y, batch_size=B_SIZE, callbacks=[])
            #model.fit((get_batch(batch_size=B_SIZE, leng=SEQLEN) for _ in range(B_CNT)))
            stats = align_stats.align_dist(model(x_test), y_test)
            print("epoch ", epoch, stats[0]/stats[1])
            #print(epoch, stats, file=f_stat)
            if epoch % 500 == 0:
                _save(model, f"models/{args.outname or args.config}.{epoch}")
            
            epoch += 1

    if True:
        print("folding")
        print(model(x_test))
        for l in model._layers[2]._layers[0]:
            try:
                l.fold_bn()
            except Exception as e:
                print(e)
                pass
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=net.CTCLoss())
        print(">>>")
        print(model(x_test))

    _save(model, f"models/{args.outname or args.config}.tflite")

if __name__ == "__main__":
    main()