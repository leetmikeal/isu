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

    # ensemble
    def command_ensemble(args):
        from ensemble import main
        main(args)

    parser_ensemble = subparsers.add_parser(
        'ensemble', help='see `-h`')
    from ensemble import setup_argument_parser
    # subcommand
    setup_argument_parser(parser_ensemble)
    parser_ensemble.set_defaults(handler=command_ensemble)

    # threshold
    def command_threshold(args):
        from threshold import main
        main(args)

    parser_threshold = subparsers.add_parser(
        'threshold', help='see `-h`')
    from threshold import setup_argument_parser
    # subcommand
    setup_argument_parser(parser_threshold)
    parser_threshold.set_defaults(handler=command_threshold)

    # collision
    def command_collision(args):
        from collision import main
        main(args)

    parser_collision = subparsers.add_parser(
        'collision', help='see `-h`')
    from collision import setup_argument_parser
    # subcommand
    setup_argument_parser(parser_collision)
    parser_collision.set_defaults(handler=command_collision)


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
