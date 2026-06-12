"""Tests for the pure display helpers in token_display (shorten, is_display_token).

clean_token is not tested here: it requires a tokenizer instance.
"""

from __future__ import annotations

from sparse_readout_prism.token_display import is_display_token, shorten


def test_shorten_collapses_internal_whitespace():
    assert shorten("a   b\tc", 10) == "a b c"


def test_shorten_passthrough_when_within_max_len():
    assert shorten("hello", 5) == "hello"  # len == max_len, no truncation


def test_shorten_truncation_boundary_keeps_max_len_minus_one():
    # 8 > 5: keep first max_len-1 (=4) chars, then append the ellipsis.
    assert shorten("abcdefgh", 5) == "abcd..."


def test_shorten_custom_ellipsis():
    assert shorten("abcdefgh", 5, ellipsis="…") == "abcd…"


def test_is_display_token_empty_and_whitespace_false():
    assert is_display_token("") is False
    assert is_display_token("   ") is False


def test_is_display_token_single_punctuation_false():
    assert is_display_token("!") is False


def test_is_display_token_single_alphanumeric_true():
    assert is_display_token("a") is True


def test_is_display_token_normal_word_true():
    assert is_display_token("hello") is True


def test_is_display_token_strips_surrounding_space():
    assert is_display_token("  word  ") is True
    assert is_display_token(" a ") is True  # stripped to a single alnum char


def test_is_display_token_pure_non_printable_false():
    assert is_display_token("\x00\x01") is False
