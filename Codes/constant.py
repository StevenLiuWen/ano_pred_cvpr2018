import os
import argparse
import configparser


def get_dir(directory):
    """
    get the directory, if no such directory, then make it.

    @param directory: The new directory.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def parser_args():
    parser = argparse.ArgumentParser(description='Options to run the network.')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the device id of gpu.')
    parser.add_argument('-i', '--iters', type=int, default=1,
                        help='set the number of iterations, default is 1')
    parser.add_argument('-b', '--batch', type=int, default=4,
                        help='set the batch size, default is 4.')
    parser.add_argument('--num_his', type=int, default=4,
                        help='set the time steps, default is 4.')

    parser.add_argument('-d', '--dataset', type=str,
                        help='the name of dataset.')
    parser.add_argument('--train_folder', type=str, default='',
                        help='set the training folder path.')
    parser.add_argument('--test_folder', type=str, default='',
                        help='set the testing folder path.')

    parser.add_argument('--config', type=str, default='training_hyper_params/hyper_params.ini',
                        help='the path of training_hyper_params, default is training_hyper_params/hyper_params.ini')

    parser.add_argument('--snapshot_dir', type=str, default='',
                        help='if it is folder, then it is the directory to save models, '
                             'if it is a specific model.ckpt-xxx, then the system will load it for testing.')
    parser.add_argument('--summary_dir', type=str, default='', help='the directory to save summaries.')
    parser.add_argument('--psnr_dir', type=str, default='', help='the directory to save psnrs results in testing.')

    parser.add_argument('--evaluate', type=str, default='compute_auc',
                        help='the evaluation metric, default is compute_auc')

    return parser.parse_args()


class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.{}".format(name))
        if not name.isupper():
            raise self.ConstCaseError('const name {} is not all uppercase'.format(name))

        self.__dict__[name] = value

    def __str__(self):
        _str = '<================ Constants information ================>\n'
        for name, value in self.__dict__.items():
            print(name, value)
            _str += '\t{}\t{}\n'.format(name, value)

        return _str


args = parser_args()
const = Const()

# inputs constants
const.DATASET = args.dataset
const.TRAIN_FOLDER = args.train_folder
const.TEST_FOLDER = args.test_folder

const.GPU = args.gpu

const.BATCH_SIZE = args.batch
const.NUM_HIS = args.num_his
const.ITERATIONS = args.iters

const.EVALUATE = args.evaluate

# network constants
const.HEIGHT = 256
const.WIDTH = 256
const.FLOWNET_CHECKPOINT = 'checkpoints/pretrains/flownet-SD.ckpt-0'
const.FLOW_HEIGHT = 384
const.FLOW_WIDTH = 512

# set training hyper-parameters of different datasets
config = configparser.ConfigParser()
assert config.read(args.config)

# for lp loss. e.g, 1 or 2 for l1 and l2 loss, respectively)
const.L_NUM = config.getint(const.DATASET, 'L_NUM')
# the power to which each gradient term is raised in GDL loss
const.ALPHA_NUM = config.getint(const.DATASET, 'ALPHA_NUM')
# the percentage of the adversarial loss to use in the combined loss
const.LAM_ADV = config.getfloat(const.DATASET, 'LAM_ADV')
# the percentage of the lp loss to use in the combined loss
const.LAM_LP = config.getfloat(const.DATASET, 'LAM_LP')
# the percentage of the GDL loss to use in the combined loss
const.LAM_GDL = config.getfloat(const.DATASET, 'LAM_GDL')
# the percentage of the different frame loss
const.LAM_FLOW = config.getfloat(const.DATASET, 'LAM_FLOW')

# Learning rate of generator
const.LRATE_G = eval(config.get(const.DATASET, 'LRATE_G'))
const.LRATE_G_BOUNDARIES = eval(config.get(const.DATASET, 'LRATE_G_BOUNDARIES'))

# Learning rate of discriminator
const.LRATE_D = eval(config.get(const.DATASET, 'LRATE_D'))
const.LRATE_D_BOUNDARIES = eval(config.get(const.DATASET, 'LRATE_D_BOUNDARIES'))


const.SAVE_DIR = '{dataset}_l_{L_NUM}_alpha_{ALPHA_NUM}_lp_{LAM_LP}_' \
                 'adv_{LAM_ADV}_gdl_{LAM_GDL}_flow_{LAM_FLOW}'.format(dataset=const.DATASET,
                                                                      L_NUM=const.L_NUM,
                                                                      ALPHA_NUM=const.ALPHA_NUM,
                                                                      LAM_LP=const.LAM_LP, LAM_ADV=const.LAM_ADV,
                                                                      LAM_GDL=const.LAM_GDL, LAM_FLOW=const.LAM_FLOW)

if args.snapshot_dir:
    # if the snapshot_dir is model.ckpt-xxx, which means it is the single model for testing.
    if os.path.exists(args.snapshot_dir + '.meta') or os.path.exists(args.snapshot_dir + '.data-00000-of-00001') or \
            os.path.exists(args.snapshot_dir + '.index'):
        const.SNAPSHOT_DIR = args.snapshot_dir
        print(const.SNAPSHOT_DIR)
    else:
        const.SNAPSHOT_DIR = get_dir(os.path.join('checkpoints', const.SAVE_DIR + '_' + args.snapshot_dir))
else:
    const.SNAPSHOT_DIR = get_dir(os.path.join('checkpoints', const.SAVE_DIR))

if args.summary_dir:
    const.SUMMARY_DIR = get_dir(os.path.join('summary', const.SAVE_DIR + '_' + args.summary_dir))
else:
    const.SUMMARY_DIR = get_dir(os.path.join('summary', const.SAVE_DIR))

if args.psnr_dir:
    const.PSNR_DIR = get_dir(os.path.join('psnrs', const.SAVE_DIR + '_' + args.psnr_dir))
else:
    const.PSNR_DIR = get_dir(os.path.join('psnrs', const.SAVE_DIR))


