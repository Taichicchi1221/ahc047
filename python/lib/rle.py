def rle(s):
    """
    Run Length Encoding
    """
    tmp, count = s[0], 1
    ret = []
    for i in range(1, len(s)):
        if tmp == s[i]:
            count += 1
        else:
            ret.append([tmp, count])
            tmp = s[i]
            count = 1
    ret.append([tmp, count])
    return ret
