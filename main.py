from brainage import BrainAgeData1, BrainAgeData2, BrainAgeModel1, BrainAgeModel2
from keras import backend as K

import experiment
import utils
import os

if __name__ == '__main__':
    args = utils.parse_args()

    if args.device == "0" or args.device == "1":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    K.set_learning_phase(0)

    if not args.with_mas:
        data = BrainAgeData1(args.work_dir)
        sess = K.get_session()
        model = BrainAgeModel1('BAG_model_part1.h5', sess, args.work_dir)
    else:
        data = BrainAgeData2(args.work_dir)
        sess = K.get_session()
        model = BrainAgeModel2('BAG_model_part1.h5', 'BAG_model_part2.h5', sess, args.work_dir)

    if args.mode == 'gsm_exp':
        attack = experiment.GsmExp(data, model, args)
    elif args.mode == 'multi_gsm':
        attack = experiment.MultiGsm(data, model, args)
    elif args.mode == 'l2_exp':
        attack = experiment.L2Exp(data, model, args)
    elif args.mode == 'l0_exp':
        attack = experiment.L0Exp(data, model, args)
    else:

        raise Exception('Invalid mode')

    attack.attack()
