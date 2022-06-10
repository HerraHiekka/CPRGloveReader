import time

from CPRState import CPRState

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--save", "-s", help="Save the session recording under given name in './recordings/'", type=str)
    parser.add_argument("--config", "-c", help="Location of the config file. Default: './'", type=str)
    args = parser.parse_args()

    if args.config is None:
        args.config = './config.yaml'

    program = CPRState(save=args.save, config=args.config)
    try:
        program.run()
    # Catch the exit signal for graceful exit and saving of current session.
    except KeyboardInterrupt:
        program.stop()


if __name__ == '__main__':
    main()
