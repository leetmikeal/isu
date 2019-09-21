# -*- coding: utf-8 -*-
import argparse
import os
import sys
import re
import ast

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


def register_training(parser):
    from training import setup_argument_parser

    def command_training(args):
        from training import main
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command_training)


def register_analyze(parser):
    from analyze.main import setup_argument_parser
    setup_argument_parser(parser)


def main():
    # top-level command line parser
    parser = argparse.ArgumentParser(prog=PACKAGE_NAME.replace(
        '_', '-'), description='isu')
    parser.add_argument('--version', action='version', version='%(prog)s ' + version)
    subparsers = parser.add_subparsers()

    # training
    parser_training = subparsers.add_parser('training', help='see `-h`')
    register_training(parser_training)

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
