"""
implicit.py
-----------

Example of using multiprocessing to execute an implicit method of a class and
store the result internally.
"""

import multiprocessing as mp
from time import sleep


class ExampleClass:
    """
    Example class with an implicit method that takes a long time to execute,
    and stores a result internally.
    """

    method_has_been_called: bool

    def __init__(self) -> None:
        self.method_has_been_called = False

    def __str__(self) -> str:
        return str(f"method_has_been_called: {self.method_has_been_called}")

    def method(self) -> None:
        """Sets method_has_been_called to True"""
        sleep(0.2)
        print("Method has been called")
        self.method_has_been_called = True


def sequential_execution(instance_list: list[ExampleClass]) -> None:
    """
    Executes the method of each ExampleClass in the list sequentially.
    """
    for example in instance_list:
        example.method()
    for example in instance_list:
        print(example)


def bad_parallel_execution(instance_list: list[ExampleClass]) -> None:
    """
    Executes the method of each ExampleClass in the list in parallel, but the
    results are not stored in the instances attributes because the method is
    executed in a different process.
    """
    with mp.Pool() as pool:
        pool.map(ExampleClass.method, instance_list)
    for example in instance_list:
        print(example)


def good_parallel_execution(instance_list: list[ExampleClass]) -> None:
    """
    Executes the method of each ExampleClass in the list in parallel and
    overwrites each instance in the list with a new instance that has been
    modified by the method.
    """
    with mp.Pool() as pool:
        instance_list = pool.map(parallel_helper, instance_list)
    for example in instance_list:
        print(example)


def parallel_helper(example: ExampleClass) -> ExampleClass:
    """
    Helper function for parallel_execution. Executes the method of the
    ExampleClass instance and returns the instance.
    """
    example.method()
    return example  # important


if __name__ == "__main__":
    print("Sequential execution")
    sequential_execution(instance_list=[ExampleClass() for _ in range(10)])

    print("\nBad parallel execution")
    bad_parallel_execution(instance_list=[ExampleClass() for _ in range(10)])

    print("\nGood parallel execution")
    good_parallel_execution(instance_list=[ExampleClass() for _ in range(10)])
