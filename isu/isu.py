# -*- coding: utf-8 -*-
import argparse
import os
import sys
import re
import ast

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from src import get_module_logger
# logger = get_module_logger(__name__)
# logger.debug("test")

# package name
PACKAGE_NAME = 'isu'
with open(os.path.join(os.path.dirname(__file__), '__init__.py')) as f:
    match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))

# adding current dir to lib path
mydir = os.path.dirname(__file__)
# sys.path.append(mydir)
sys.path.insert(0, mydir)


def register_training_2d(parser):
    from training_2d import setup_argument_parser

    def command_training_2d(args):
        from training_2d import main
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command_training_2d)

def register_predict_2d(parser):
    from predict_2d import setup_argument_parser

    def command_predict_2d(args):
        from predict_2d import main
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command_predict_2d)

def register_training_3d(parser):
    from training_3d import setup_argument_parser

    def command_training_3d(args):
        from training_3d import main
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command_training_3d)

def register_predict_3d(parser):
    from predict_3d import setup_argument_parser

    def command_predict_3d(args):
        from predict_3d import main
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command_predict_3d)


def register_analyze(parser):
    from analyze.main import setup_argument_parser
    setup_argument_parser(parser)


def main():
    # top-level command line parser
    parser = argparse.ArgumentParser(prog=PACKAGE_NAME.replace(
        '_', '-'), description='isu')
    parser.add_argument('--version', action='version', version='%(prog)s ' + version)
    subparsers = parser.add_subparsers()

    # training_2d
    parser_training_2d = subparsers.add_parser('training-2d', help='see `-h`')
    register_training_2d(parser_training_2d)

    # predict_2d
    parser_predict_2d = subparsers.add_parser('predict-2d', help='see `-h`')
    register_predict_2d(parser_predict_2d)

    # training_3d
    parser_training_3d = subparsers.add_parser('training-3d', help='see `-h`')
    register_training_3d(parser_training_3d)

    # predict_3d
    parser_predict_3d = subparsers.add_parser('predict-3d', help='see `-h`')
    register_predict_3d(parser_predict_3d)

    # analyze
    parser_analyze = subparsers.add_parser('analyze', help='see `-h`')
    register_analyze(parser_analyze)

    # to parse command line arguments, and execute processing
    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        # if unknwon subcommand was given, then showing help
        parser.print_help()


if __name__ == '__main__':
    main()
