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


if __name__ == '__main__': main()