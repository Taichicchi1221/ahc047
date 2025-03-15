def base_to_n(X, n):
    if (int(X/n)):
        return base_to_n(int(X/n), n)+str(X % n)
    return str(X % n)
