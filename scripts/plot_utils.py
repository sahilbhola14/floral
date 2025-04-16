def to_latex_sci(x, precision=2):
    s = f"{x:.{precision}e}"
    base, exp = s.split("e")
    return r"%s \times 10^{%d}" % (base, int(exp))
