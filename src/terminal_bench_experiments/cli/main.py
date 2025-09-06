from typer import Typer

from terminal_bench_experiments.cli.db import db_app

app = Typer(no_args_is_help=True)
app.add_typer(db_app, name="db", help="Database operations")


def main():
    app()


if __name__ == "__main__":
    main()
