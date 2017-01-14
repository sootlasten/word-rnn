import argparse
import cPickle as pickle

from model import Network


def load_model(file_path):
    """Loads the network from file."""
    with open(file_path, 'rb') as f:
        d = pickle.load(f)

    net = Network(d['n_h_layers'], d['n_h_units'],
                  1, d['cell_type'], d['vocab_size'])

    return net, d['vocab_dict'], d['reverse_vocab_dict']


def parse_args():
    parser = argparse.ArgumentParser(description='Sample from the network.')
    parser.add_argument('--info_file', help='Path to the file that contains '
                                             'the model information')
    parser.add_argument('--ckpts_dir', help='Path to the directory '
                                                  'checkpoints are saved to.',
                        default='ckpts/')
    parser.add_argument('--prime_text', help='Text to feed the network with before sampling',
                        default=None)
    parser.add_argument('--length', help='Number of words to sample. Defaults to 100.',
                        type=int, default=100)
    parser.add_argument('--out_file', help='Path to output file. Defaults to ./sample.txt',
                        default='sample.txt')
    return parser.parse_args()


args = parse_args()
net, vocab_dict, reverse_vocab_dict = load_model(args.info_file)

with open(args.out_file, 'w') as f:
    sample_text = net.sample(args.length, args.prime_text, vocab_dict,
                             reverse_vocab_dict, args.ckpts_dir)
    f.write(sample_text)
