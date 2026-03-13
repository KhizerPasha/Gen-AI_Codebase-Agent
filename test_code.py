# sample_code.py


# ── 1. FIBONACCI — for EXPLAIN query ──────────────────────
def fibonacci(n):
    """Returns the nth Fibonacci number using recursion."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


# ── 2. BUGGY FUNCTION — for BUG FINDER query ──────────────
def get_average(numbers):
    """Returns the average of a list of numbers."""
    total = 0
    for i in range(len(numbers) + 1):   # BUG 1: off-by-one → IndexError
        total += numbers[i]

    average = total / len(numbers)       # BUG 2: ZeroDivisionError if list empty
    return average

    print(f"Average is: {average}")      # BUG 3: unreachable dead code


# ── 3. UNOPTIMAL FUNCTION — for REFACTOR query ─────────────
def get_even_squares(numbers):
    """Returns squares of even numbers from a list."""
    result = []
    for i in range(len(numbers)):
        if numbers[i] % 2 == 0:
            square = numbers[i] * numbers[i]
            result.append(square)
    return result
