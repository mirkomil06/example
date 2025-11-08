from math import isqrt


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    limit = isqrt(n)
    candidate = 5
    while candidate <= limit:
        if n % candidate == 0 or n % (candidate + 2) == 0:
            return False
        candidate += 6
    return True


def main() -> None:
    raw = input("Enter an integer to test for primality: ")
    try:
        value = int(raw)
    except ValueError:
        print("Input must be an integer.")
        return

    if is_prime(value):
        print(f"{value} is prime.")
    else:
        print(f"{value} is not prime.")


if __name__ == "__main__":
    main()
