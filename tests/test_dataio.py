import pytest

from pinpai.dataio import parse_table


def test_zero_and_negative_freq_rows_are_reported():
    csv = "freq_hz,zreal,zimag\n0,1,2\n-10,3,4\n1,100,-5\n100,90,-40\n"
    out = parse_table(csv)
    assert len(out["freqs"]) == 2
    lines = {e["line"]: e["reason"] for e in out["excluded"]}
    assert lines[2].startswith("频率 0")
    assert lines[3].startswith("频率 -10")
    # 无表头时行号从 1 起
    out2 = parse_table("1,100,-5\n0,1,2")
    assert {e["line"] for e in out2["excluded"]} == {2}


def test_non_numeric_and_missing():
    out = parse_table("f,re,im\n1,100,-5\nx,1,2\n2,50")
    assert len(out["freqs"]) == 1
    assert out["excluded"][0]["line"] == 3


def test_all_invalid_raises():
    with pytest.raises(ValueError):
        parse_table("freq_hz,zreal,zimag\n0,1,2")


def test_aliases_and_tsv():
    out = parse_table("频率\t实部\t虚部\n1\t100\t-5\n10\t90\t-30")
    assert len(out["freqs"]) == 2
