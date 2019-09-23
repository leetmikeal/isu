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

    # connection
    def command_connection(args):
        from connection import main
        main(args)

    parser_connection = subparsers.add_parser(
        'connection', help='see `-h`')
    from connection import setup_argument_parser
    # subcommand
    setup_argument_parser(parser_connection)
    parser_connection.set_defaults(handler=command_connection)

    # pxcm
    def command_pxcm(args):
        from pxcm import main
        main(args)

    parser_pxcm = subparsers.add_parser(
        'pxcm', help='see `-h`')
    from pxcm import setup_argument_parser
    # subcommand
    setup_argument_parser(parser_pxcm)
    parser_pxcm.set_defaults(handler=command_pxcm)


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
