"""Test basic functionality of the package."""


def test_import():
    """Test that the package can be imported."""
    import recsys  # noqa: F401


def test_version():
    """Test that the version is defined."""
    from recsys import __version__
    assert isinstance(__version__, str)
    assert len(__version__.split('.')) == 3  # Major.Minor.Patch
