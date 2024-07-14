from metrics import MetricX

src = [
    "The quick brown fox jumps over the lazy dog.",
    "An apple a day keeps the doctor away."
]
pred = [
    "Le renard brun rapide saute par-dessus le chien paresseux.",
    "Une pomme par jour éloigne le médecin."
]
ref = [
    ["Le rapide renard brun bondit par-dessus le chien endormi."],
    ["Manger une pomme chaque jour éloigne le docteur."]
]

def main():
    metric = MetricX(variation='metricx', size='l', batch_size=2)

    res = metric(src=src, pred=pred, ref=ref)

    print(res)

    return res


def test_main():
    res = main()
    assert all(round(a, 10) == round(e, 10) for a, e in zip(res, [3.2230827808380127, 2.6962897777557373])), res


if __name__ == '__main__': main()