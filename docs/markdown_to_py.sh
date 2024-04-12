#!/usr/bin/env -S bash

sed -n '/^```python/,/^```/ p' < $1 | sed '/^```/ d' > $2
