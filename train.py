import argparse

from model import Network


def parse_args():
    parser = argparse.ArgumentParser(description='Train the network.')
    parser.add_argument('--ckpts_dir', help='Path to the directory to save '
                                                  'checkpoints to.',
                        default='ckpts/')
    parser.add_argument('--params_file', help='Path to the file where to read model '
                                              'params from.',
                        default='params.txt')
    return parser.parse_args()


args = parse_args()

with open(args.params_file, 'r') as f:
    params = f.readlines()

p_dict = dict()
for param in params:
    p = param.rstrip('\n').split('=')
    try:
        p_dict[p[0]] = int(p[1])
    except ValueError:
        try:
            p_dict[p[0]] = float(p[1])
        except ValueError:  # cell type
            p_dict[p[0]] = p[1]

net = Network(
    n_h_layers=p_dict.get('n_h_layers', 2),
    n_h_units=p_dict.get('n_h_units', 100),
    seq_length=p_dict.get('seq_length', 25),
    cell_type=p_dict.get('cell_type', 'LSTM'),
    vocab_size=p_dict.get('vocab_size', 20000))

net.train(
    batch_size=p_dict.get('batch_size', 40),
    eta=p_dict.get('eta', 0.001),
    grad_clip=p_dict.get('grad_clip', 5),
    n_epochs=p_dict.get('n_epochs', 10),
    train_frac=p_dict.get('train_frac', 0.95),
    checkpoint_dir=args.ckpts_dir)
