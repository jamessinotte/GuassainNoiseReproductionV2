"""Strict source scoring and separately labeled corrected diagnostics."""
import re
import string


def source_float(value):
    # Pinned Inspect supports Unicode superscripts such as 10², not just floats.
    superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    for i, char in enumerate(value):
        if char in superscripts:
            base = float(value[:i]) if i else 1.
            exponent = int(value[i:].translate(str.maketrans(superscripts, "0123456789")))
            return base ** exponent
    return float(value)


def legacy_last_number(text):
    for word in reversed(text.split()):
        cleaned = word.strip('.,!?)(]}')
        if cleaned.replace('-', '').replace('.', '').isdigit():
            return cleaned
    return ""


def dollar_answer(text):
    matches = re.findall(r'\$([^$]*?[-+]?\d*\.?\d+[^$]*?)\$', text)
    if not matches:
        return None
    text = matches[-1].strip().replace(",", "")
    fraction = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', text)
    try:
        if fraction:
            return float(fraction[1]) / float(fraction[2])
        number = re.search(r'[-+]?\d*\.?\d+', text)
        return float(number.group()) if number else float(text)
    except (ValueError, ZeroDivisionError):
        return None


def inspect_numeric_end(text, target):
    # Deliberately preserves pinned Inspect's endswith comparison: 172 matches 72.
    value, expected = text.strip().casefold(), target.strip().casefold()
    if expected.isnumeric():
        value = re.sub(r"[$,£,€]", "", value)
        value = re.sub(r"\.(?=\s|$|\D)", "", value)
        words = list(reversed(re.split(r"\s+", value)))
        value = next((w for w in words if w.replace(".", "").isnumeric()), words[0])
        if value.replace(".", "").isnumeric():
            value = format(source_float(value), ".5g")
        expected = format(source_float(expected), ".5g")
    else:
        value = value.strip(string.whitespace + string.punctuation)
        expected = expected.strip(string.whitespace + string.punctuation)
    return value, value.endswith(expected)


def score(output, target, method, full_text=None):
    if method == "next_token_exact":
        return output, output == target
    if method == "next_token_numeric_key_corrected":
        repaired = {"1": "A", "2": "B", "3": "C", "4": "D"}.get(str(target), target)
        return output, output == repaired
    if method == "legacy_cot":
        answer = legacy_last_number(full_text if full_text is not None else output)
        return answer, answer == target
    if method == "continuation_number_corrected":
        answer = legacy_last_number(output)
        return answer, answer == target
    if method == "dollar":
        answer = dollar_answer(output)
        return answer, answer is not None and abs(answer - float(target)) < 1e-6
    if method == "inspect_numeric_end":
        return inspect_numeric_end(output, target)
    if method == "numeric_exact_corrected":
        # Do not reuse the faithful scorer's rounding/suffix/sign handling.
        matches = re.findall(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹]*", output)
        value = matches[-1].replace(",", "") if matches else ""
        try:
            return value, abs(source_float(value) - source_float(str(target).replace(",", ""))) < 1e-6
        except ValueError:
            return value, False
    raise ValueError(f"Unknown scoring method {method}")
