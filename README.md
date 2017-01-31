# walkcompare

A way of using Multiple Correspondence analysis to analyse Web Archives

Walk-Compare is a class developed for the [Web Archives for Longitudinal Knowledge Project (WALK)](https://github.com/web-archive-group/WALK) as a way to observe multiple Web Archive collections. However, it is more suitable a wrapper for the Matplotlib-venn and the mca libraries. It takes a list of lists and turns it into a graph that maps the collections according to what items they have in common.

To install:

.. code-block:: bash

    pip install walkcompare


Example:

```python

from walkcompare import compare
import json
with open('walkcompare/data/parliamentary_committees.json') as f:

""" {
    "INAN":{
        "name": "Indigenous and Northern Affairs",
        "membership":["C Warkentin", "J Crowder", "C Bennett", "S Ambler", "D Bevington", "R Boughen", "R Clarke", "J Genest-Jourdain", "J Hillyer", "C Hughes", "G Rickford", "K Seeback"]
    },

    "HUMA":{
        "name":"Human Resources, Skills and Social Development and the Status of Persons with Disabilities",
        "membership": ["E Komarnicki", "C Charlton", "R Cuzner", "M Boutin-Sweet", "B Butt", "R Cleary", "J Daniel", "F Lapointe", "K Leitch", "C Mayes", "P McColeman", "D Shory"]
    },
    ...
    }
"""

    data = json.load(f)
values = [y['membership'] for  y in data.values()]
names = [q for q in data.keys()]
print(names)
compare = compare(values, names)
compare.LABEL_BOTH_FACTORS = True
compare.adjust = True
compare.plot_ca()
```

This produces:

![Parliamentary](/walkcompare/examples/parl_comm.png)
