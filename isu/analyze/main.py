import argparse
import os
import sys

# adding current dir to lib path
mydir = os.path.dirname(__file__)
sys.path.insert(0, mydir)

def setup_argument_parser(parser):
    """
    Set argument
    """
    subparsers = parser.add_subparsers()

    # last_precision
    def command_last_precision(args):
        from last_precision import main
        main(args)

    parser_last_precision = subparsers.add_parser(
        'last-precision', help='see `-h`')
    from last_precision import setup_argument_parser
    # subcommand
    setup_argument_parser(parser_last_precision)
    parser_last_precision.set_defaults(handler=command_last_precision)


if __name__ == '__main__':
    # setup
    parser = argparse.ArgumentParser()
    setup_argument_parser(parser)
    args = parser.parse_args()

    if hasattr(args, 'handler'):
        args.handler(args)
        args.verbose
        print('')
    else:
        # if unknwon subcommand was given, then showing help
        parser.print_help()
