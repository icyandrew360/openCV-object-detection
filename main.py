import argparse

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-p', '--prototxt', required=True, help='path to prototxt file')
    parse.add_argument('-m', '--model', required=True, help='path to the Caffe pre-trained model')
    parse.add_argument('-c', '--confidence', type=float, default=0.2, help='minimum probability to filter weak detections')
    args = vars(parse.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)