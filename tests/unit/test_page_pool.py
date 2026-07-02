import pytest

from kairyu.engine.core.pages import PagePool


def test_allocates_distinct_pages():
    pool = PagePool(num_pages=4)
    pages = pool.allocate(3)
    assert len(pages) == 3
    assert len(set(pages)) == 3
    assert pool.num_free == 1


def test_free_returns_pages_for_reuse():
    pool = PagePool(num_pages=2)
    pages = pool.allocate(2)
    pool.free(pages)
    assert pool.num_free == 2
    assert len(pool.allocate(2)) == 2


def test_exhaustion_raises():
    pool = PagePool(num_pages=2)
    pool.allocate(2)
    with pytest.raises(MemoryError, match="pages"):
        pool.allocate(1)


def test_double_free_rejected():
    pool = PagePool(num_pages=2)
    pages = pool.allocate(1)
    pool.free(pages)
    with pytest.raises(ValueError, match="not allocated"):
        pool.free(pages)


def test_zero_allocation_is_empty():
    pool = PagePool(num_pages=1)
    assert pool.allocate(0) == ()
