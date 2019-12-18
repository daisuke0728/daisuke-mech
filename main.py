import argparse

from hyperparam import tune_hyperparams
from nas import nas
from preprocess import select_preprocess
from task import Task


def main():
    '''Auto Kaggle Solver
    Given task specifications, determine optimal preprocess, network architecture, and hyperparameters automatically.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, help='Task name')
    #parser.add_argument('--gpu', '-g', type=int,nargs='?', help='GPU ids to use')

    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id, split with ","')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')

    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    args = parser.parse_args()

    # Instantiate task object
    task = Task(args.task)

    # Select optimal preprocess
    print('Start preprocess selection.')
    preprocess_func = select_preprocess(args, task)
    print('Finished.')

    # Design optimal network architecture                                                                            
    print('Start Network Architecture search.')
    model = nas(args, task, preprocess_func)
    print('Finished.')

    # Tune hyperparamters                                                                                            
    print('Start hyperparameter tuning.')
    hyperparams = tune_hyperparams(args, task, preprocess_func, model)
    print('Finished.')

    return preprocess_func, model, hyperparams

if __name__ == '__main__':
    main()

