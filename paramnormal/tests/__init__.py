from pkg_resources import resource_filename


def test(*args):
    try:
        import pytest
    except ImportError:
        raise ImportError("`pytest` is required to run the test suite")

    options = [resource_filename("paramnormal", "tests")]
    options.extend(list(args))
    return pytest.main(options)
