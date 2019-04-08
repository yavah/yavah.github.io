---
layout: post
title: Automating consumption of SDMX files using Python
tags: [SDMX, Python]
excerpt_separator: <!--more-->
---
This notebook is to describe how to create a function in Python to consume SDMX files. The data can be then converted to Excel files or save in a database.

<!--more-->

```python
from pandasdmx import Request
import pandas as pd
import csv
import os
import warnings
warnings.filterwarnings('ignore')

```


```python
def sdmx(country, SDMXurl):
     
    '''Create function to convert SDMX files to Excel'''
   
    IMF = Request ("IMF_SDMXCENTRAL")
    data_resp = IMF.data(url=SDMXurl)
    data_frame = data_resp.write(data_resp.data.series, dtype = str, asframe = True, parse_time=False)
    attributes_frame = data_resp.write(data_resp.data.series, dtype = str, asframe = True, attributes = "s", parse_time=False)
    
    #One time only. Once the the ECOFIN_CL is pulled from SDMX Central it could be used for all data categories.
    IMF_flow = IMF.dataflow('CPI')
    dsd = IMF_flow.dataflow.CPI.structure()
    ECOFIN_CL = IMF_flow.write().codelist.loc['CL_INDICATOR']


    data = data_frame.transpose()
    data = data.reset_index()
    data = pd.merge(data, ECOFIN_CL, how = 'left', left_on = "INDICATOR",  right_index=True)

    data.insert(3, "DESCRIPTOR", data.name)
    data.drop(labels = "name", axis = 1, inplace = True)

    #Save to Excel but it could also be pushed to a database
    wd = os.getcwd()
    data.to_excel('%s.xlsx' % country)
```


```python
file=[]
f = open('SDMXlinks.csv')
csv_f = csv.reader(f)
next(csv_f)
for row in csv_f:
    file.append(row)
    
```


```python
for country, SDMXurl in file:
    sdmx(country, SDMXurl)
```
