"""Report short text-block endings from pdftotext -bbox-layout output."""
import sys
import re
import xml.etree.ElementTree as ET

with open(sys.argv[1]) as source:
    xml = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', source.read())
root = ET.fromstring(xml)
ns = {'x': 'http://www.w3.org/1999/xhtml'}
for page_number, page in enumerate(root.findall('.//x:page', ns), 13):
    for block in page.findall('.//x:block', ns):
        lines = block.findall('x:line', ns)
        if not lines:
            continue
        line = lines[-1]
        words = line.findall('x:word', ns)
        value = ' '.join(w.text or '' for w in words)
        left, right = float(line.get('xMin')), float(line.get('xMax'))
        if 100 < left < 125 and right < 320 and float(line.get('yMin')) > 60 and len(value) > 1:
            print(f'{page_number}: x={right:.0f}, y={float(line.get("yMin")):.0f}: {value}')
