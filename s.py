def isalpha(s):
    """Check if all characters in the string are alphabetic (unicode)."""
    for ch in s:
        if not ch.isalpha():
            return False
    return True

print(isalpha("|")) 