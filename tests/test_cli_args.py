from crop_datasets.cli import build_parser


def test_root_option_after_subcommand_is_supported():
    parser = build_parser()
    args = parser.parse_args(["list", "--root", "./data"])
    assert args.root == "./data"
    assert args.cmd == "list"


def test_root_option_before_subcommand_is_supported():
    parser = build_parser()
    args = parser.parse_args(["--root", "./data", "list"])
    assert args.root == "./data"
    assert args.cmd == "list"
