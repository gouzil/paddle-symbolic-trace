import unittest

from test_case_base import TestCaseBase

import paddle


def foo(y: paddle.Tensor):
    def local(z):
        return y + z

    return local(1)


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()


# Instructions:
# LOAD_CLOSURE
# LOAD_DEREF
# LOAD_CLASSDEREF
# STORE_DEREF
# DELETE_DEREF
