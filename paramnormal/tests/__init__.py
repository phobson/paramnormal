from pathlib import Path


def test(*args):
    try:
        import pytest
    except ImportError:
        raise ImportError("`pytest` is required to run the test suite")

    options = [str(Path(__file__).parent)]
    options.extend(list(args))
    return pytest.main(options)
