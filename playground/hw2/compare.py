def pprint(text):
    for l in text:
        print(l)
    print("=" * 100, '\n')

baseline_fp, baseline_fn = set(), set()
baseline = baseline_fp
with open('baseline_false_pred.txt') as f:
    for l in f:
        if l == '\n':
            baseline = baseline_fn
            continue
        l = l.strip()
        baseline.add(l[1:-1])

# print(baseline_fn)
# print(baseline_fp)

new_feature_fp, new_feature_fn = set(), set()
new_feature = new_feature_fp
with open('new_feature_false_pred.txt') as f:
    for l in f:
        if l == '\n':
            new_feature = new_feature_fn
            continue
        l = l.strip()
        new_feature.add(l[1:-1])

# pprint(baseline_fp)
# pprint(baseline_fn)
# pprint(new_feature_fp)
# pprint(new_feature_fn)

baseline = (baseline_fp | baseline_fn)
new_feature = (new_feature_fn | new_feature_fp)

both = baseline & new_feature

better = baseline - both
print("Should be negative")
pprint(baseline_fp & better)

print("Should be positive")
pprint(baseline_fn & better)

worse = new_feature - both
print("Should be negative")
pprint(worse & new_feature_fp)
print("Should be positive")
pprint(worse & new_feature_fn)


# print("Both wrong:")
# pprint(list(both))
# print("="*100)
# print("Better:")
# pprint(list(better))
# print("="*100)
# print("Worse:")
# pprint(list(worse))