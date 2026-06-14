import re, collections

tex = open(
    'D:/hachaton/Elliptic-Bitcoin-Anomaly-Detection/Final-report/elliptic_report.tex',
    encoding='utf-8'
).read()

begins = re.findall(r'\\begin\{(\w+)\}', tex)
ends   = re.findall(r'\\end\{(\w+)\}',   tex)
b, e   = collections.Counter(begins), collections.Counter(ends)
issues = [(k, b[k], e[k]) for k in sorted(set(b) | set(e)) if b[k] != e[k]]
print('=== Environment mismatches ===')
print('None' if not issues else '\n'.join(f'  {k}: begin={bc} end={ec}' for k,bc,ec in issues))

cites   = {c.strip() for m in re.findall(r'\\cite\{([^}]+)\}', tex) for c in m.split(',')}
bibs    = set(re.findall(r'\\bibitem\{([^}]+)\}', tex))
missing = sorted(cites - bibs)
print('\n=== Missing bibitem entries ===')
print('None' if not missing else '\n'.join(f'  {m}' for m in missing))

labels  = set(re.findall(r'\\label\{([^}]+)\}', tex))
used    = set(re.findall(r'\\ref\{([^}]+)\}', tex))
undef   = sorted(used - labels)
print('\n=== Undefined \\ref targets ===')
print('None' if not undef else '\n'.join(f'  {u}' for u in undef))

print('\n=== Figure count ===')
figs = re.findall(r'\\includegraphics[^{]*\{([^}]+)\}', tex)
print(f'  {len(figs)} figures referenced')
