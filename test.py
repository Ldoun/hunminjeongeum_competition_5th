import os
import argparse


if __name__ == "__main__":
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default="0")
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--total_epoch", type=int, default=40)
    parser.add_argument("--warmup_step", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--reload_from", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--valid_every", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--strategy", type=str, default="step")
    parser.add_argument("--max_vocab_size", type=int, default=-1)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--session", type=str)

    args = parser.parse_args()

    with open('ssss.txt', 'w') as writer:
        writer.write('aaaaaaaaaaa')

    os.system('cat ssss.txt')

    print(os.listdir('/tmp/kenlm/'))

'''b = ['a','b']
with open('file.txt','w') as a:
    for i in b:
        a.write(i + '\n')'''