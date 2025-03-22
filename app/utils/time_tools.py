from datetime import datetime

from dateutil import parser as date_parser


def parse_time(time_str: str) -> datetime:
    return date_parser.parse(time_str.strip())


def transfer_time_to_str(time_str: str) -> str:
    return date_parser.parse(time_str.strip()).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    import rich

    current_time = "2020/1/1 12:00:00"
    rich.print(parse_time(current_time))
    rich.print(transfer_time_to_str(current_time))


if __name__ == "__main__":
    main()
