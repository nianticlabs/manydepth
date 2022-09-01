import glob
from manydepth.test_simple import test_simple, _parse_args


def parse_args_multiple():
    parser = _parse_args()
    parser.add_argument("--dir", type=str, help="Data in dir should be sorted by their names")
    return parser.parse_args()


def set_source_target(args, source, target):
    args.source_image_path = source
    args.target_image_path = target
    return args


if __name__ == '__main__':
    args = parse_args_multiple()
    files = sorted(glob.glob(args.dir + "/*"))
    for source, target in zip(files[:-1], files[1:]):
        args = set_source_target(args, source, target)
        test_simple(args)
