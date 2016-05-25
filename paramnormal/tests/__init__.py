from pkg_resources import resource_filename

import pytest

import paramnormal

def test(*args):
    options = [resource_filename('paramnormal', 'tests')]
    options.extend(list(args))
    return pytest.main(options)