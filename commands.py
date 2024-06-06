import argparse
from anomaly_detection.train import train
from anomaly_detection.infer import infer


def add_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('-output', default='model/default.pt', type=str)
    return parser


def add_infer_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('-timedim', default=300, type=int)
    parser.add_argument('-model', default='Adam_batch_size=32;epoch=30;BGL;tf-idf.pt', type=str)
    parser.add_argument('-time', default='model/time_embedding.pt', type=str)
    return parser


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('stage', help="")
    parser.add_argument('-encoding', default='hdfs_sentence2vec.pkl', type=str)
    parser.add_argument('-embeddim', default=300, type=int)
    parser.add_argument('-timedim', default=300, type=int)
    parser.add_argument('-dataset', default='hdfs', type=str)
    parser.add_argument('-type', default=10, type=str)
    parser.add_argument('-bi', default='True', type=bool)
    parser.add_argument('-attn', default='True', type=bool)
    parser.add_argument('-indir', default='data/LogInsight/hdfs', type=str)
    return parser


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = add_common_arguments(parser)

    args = parser.parse_args()
    if args.stage == 'train':
        parser = add_train_arguments(parser)
    elif args.stage == 'infer':
        parser = add_infer_arguments(parser)
    args = parser.parse_args()

    if args.stage == 'train':
        print('training')
        train(args)
    elif args.stage == 'infer':
        infer(args)


if __name__ == "__main__":
    parse_arguments()