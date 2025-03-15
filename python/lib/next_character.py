
def next_character(c):
    return chr((ord(c) - ord("a") + 1) % 26 + ord("a"))
