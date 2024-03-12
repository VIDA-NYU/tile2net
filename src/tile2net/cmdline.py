from __future__ import annotations
from typing import Any

import argparse
from argparse import ArgumentParser


class Arg:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, parser: argparse.ArgumentParser):
        parser.add_argument(*self.args, **self.kwargs)


class Args:
    __call__ = ArgumentParser.add_argument

    def call(self, *args, **kwargs):
        return Arg(*args, **kwargs)

    locals()['__call__'] = call


arg = Args()


def dispatch_command(function: Arg | Any) -> None:
    parser = argparse.ArgumentParser()
    for arg in function.args:
        arg(parser)  # This will call the __call__ method of each Arg instance
    args = parser.parse_args()
    function(args)


def dispatch_commands(*functions: Arg | Any) -> None:
    parser = argparse.ArgumentParser(description="Command line utility.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Create a subparser for each command
    for function in functions:
        command_name = function.__name__
        command_parser = subparsers.add_parser(command_name, help=function.__doc__)
        for arg_instance in function.args:
            arg_instance(command_parser)  # Add arguments to the subparser
        command_parser.set_defaults(func=function)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


# Update the Commandline class to optionally accept a name for the command
class Commandline:
    def __init__(self, *args: arg, name: str = None):
        self.args = args
        # self.name = name

    def __call__(self, function):
        function.args = self.args  # Attach args to the function
        self.name = function.__name__
        return function


class Namespace(argparse.Namespace):
    name: str
    location: str


if __name__ == '__main__':
    args_greeting = [
        arg('--name', '-n', type=str, required=True),
        arg('--location', '-l', type=str, required=True),
    ]
    args_farewell = [
        arg('--name', '-n', type=str, required=True),
    ]
    commandline_greeting = Commandline(*args_greeting)
    commandline_farewell = Commandline(*args_farewell)


    @commandline_greeting
    def greet(args: Namespace) -> None:
        print(f'Hello, {args.name} from {args.location}!')


    @commandline_farewell
    def farewell(args: Namespace) -> None:
        print(f'Goodbye, {args.name}!')


    # dispatch_commands(greet, farewell)
    dispatch_command(greet)
