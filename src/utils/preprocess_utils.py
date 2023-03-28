# Punctuation removal
def leave_only_alpha(textcol):
    return textcol.apply(lambda x: "".join(e for e in x if (e.isalpha() or e == " ")))


# Lowercasing with special token
def special_lowercase(x):
    chars = []
    for char in x:
        if char.lower() != char:
            chars.append("# ")
            chars.append(char.lower())
        else:
            chars.append(char.lower())
    return "".join(chars)


# Word Count Feature
def feature_wordcount(x):
    length = len(x.split())
    if length < 5:
        return "+ " + x
    elif 5 <= length < 10:
        return "++ " + x
    else:
        return "+++ " + x


def preprocess_text(textcol):
    # textcol = leave_only_alpha(textcol)
    textcol = textcol.apply(special_lowercase)
    # textcol = textcol.str.lower()
    # textcol = textcol.apply(feature_wordcount)
    return textcol






