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
    parser.add_argument("--outname")
    args = parser.parse_args()
    
    # Prepare model
    schedule = schedules.schedules[args.schedule]
    net_cfg = cfg.configs[args.config]()

    model = net.make_model(net_cfg)
    #model.compile(optimizer=tf.keras.optimizers.Adam(), loss=net.CTCLoss())

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=net.CTCLoss())

    # load files
    X, Y = data_loader.load_dir(args.data_dir)

    rng = random.Random(4747)
    def get_batch(*, batch_size=100, leng=5000):
        x, y = data_loader.prep_batch(X, Y, rng, batch_size, leng)
        res = (
            tf.constant(x),
            tf.constant(y, dtype=tf.int32)
        )
        return res

    x_test, y_test = get_batch(batch_size=100, leng=5000)
    print(x_test[0])
    epoch = 1

    def _save(model, fname):
        quant = converter.convert(model, get_batch)
        with open(fname, "wb") as f:
            f.write(quant)
        print(fname, len(quant)) 

    f_stat = open(f"models/{args.outname or args.config}.stat", "w")

    for (n_epochs, lr) in schedule:
        tf.keras.backend.set_value(model.optimizer.lr, lr)
        for _ in range(n_epochs):
            B_SIZE = 15
            B_CNT = 1500 // B_SIZE
            model.fit(get_batch(batch_size=B_SIZE, leng=5000) for _ in range(B_CNT))
            
            stats = align_stats.align_dist(model(x_test), y_test)
            print("epoch ", epoch, stats[0]/stats[1])
            epoch += 1
            print(epoch, stats, file=f_stat)
            if epoch % 100 == 0:
                _save(model, f"models/{args.outname or args.config}.{epoch}")

    _save(model, f"models/{args.outname or args.config}.tflite")

if __name__ == "__main__":
    main()