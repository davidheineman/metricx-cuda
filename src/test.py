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

    print('Running...')

    res = metric(src=src, pred=pred, ref=ref)

    print(res)

    return res


def test_main():
    res = main()
    # On CUDA: [3.2230851650238037, 2.6962885856628420]
    # On MPS:  [3.2230827808380127, 2.6962897777557373]
    assert all(round(a, 5) == round(e, 5) for a, e in zip(res, [3.2230827, 2.6962897])), res


if __name__ == '__main__': main()