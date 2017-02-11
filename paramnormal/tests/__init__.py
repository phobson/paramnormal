from pkg_resources import resource_filename

import pytest

import paramnormal


def test(*args):
    try:
        import pytest
    except ImportError as e:
        raise ImportError("`pytest` is required to run the test suite")

    options = [resource_filename('paramnormal', 'tests')]
    options.extend(list(args))
    return pytest.main(options)
