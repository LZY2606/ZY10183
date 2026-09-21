from eis.dataio import parse_table

SAMPLE = """freq_hz,z_re,z_im,sigma_re,sigma_im
# a comment
100,110,-5,0.5,0.5
0,100,-50,1,1
10,95,-40,0.4,0.4
-3,90,-30,1,1
bad,x,y
1000,105,-2,0.2,0.2
"""


def test_nonpositive_excluded_with_line_numbers():
    tab = parse_table(SAMPLE)
    assert tab["valid_count"] == 3
    excluded = tab["excluded"]
    reasons = {(r["line"], r["reason"].split("(")[0]) for r in excluded}
    assert (4, "频率非正") in reasons
    assert (6, "频率非正") in reasons
    assert any("无法解析" in r["reason"] for r in excluded)
    # 原始数据序号仍然连续
    ords = [r["rownum"] for r in tab["rows"]]
    assert ords == list(range(1, 7))


def test_valid_sorted_ascending_freq():
    tab = parse_table(SAMPLE)
    freqs = [r["freq"] for r in tab["valid"]]
    assert freqs == sorted(freqs)
    assert freqs[0] > 0


def test_whitespace_and_headerless():
    txt = "10 110 -5\n100 105 -2\n"
    tab = parse_table(txt)
    assert tab["valid_count"] == 2


def test_sigma_default_and_positivity():
    tab = parse_table("f,re,im\n1,1,1\n2,2,2\n")
    assert tab["valid"][0]["sigma_re"] == 1.0
    bad = parse_table("f,re,im,sr\n1,1,1,-2\n")
    assert bad["valid_count"] == 0
    assert "sigma" in bad["excluded"][0]["reason"]
