import os
import traceback
from config import get_configs
import train
import train_jocor
import train_rcml

def main():
    args = get_configs()
    print(args, '\n')
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    print('###### Train start ######')
    if args.scheme == 'JoCoR':
        train_jocor.run_train(args)
    elif args.scheme == 'RCML':
        train_rcml.run_train(args)
    else:
        train.run_train(args)
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
