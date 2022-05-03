# Documentation
This document will explain how to use the different documents of the directory.



## PeriodiqueGarbreVisualisation_rel.py


This script is for visualisation of patterns with relative times. (number not timestamp)

###Requirement
To use this script, you need a pattern with the output format of Esther bitbucket code. You need to print the shift in the output.

Using this command line will work: run_mine.py _file_path_ --print-details -D shift (see official documentation for more details)


###Launch PeriodiqueGarbreVisualisation_rel.py
Then select the pattern in your output file and paste it as the parameter of the script in a terminal

After you can launch the script.


In order to run one of the visualisation script you need the folowing packages:

- dash (pip install dash)

- dash_cytoscape (pip install dash-cytoscape)

- plotly.express (should be install with dash if not: pip install plotly_express)

- pandas (pip install pandas)

Pachkages can also be found in IDE package explorer or with conda.



## PeriodiqueGarbreVisualisation_abs.py


This script is for visualisation of patterns with absolute times. (timestamp)


###Requirement
To use this script, you need a pattern with the output format of Esther bitbucket code. You need to print the real and ideal time in the output.

Using this command line will work:

run_mine.py file_path --print-details -T _first_time_in_dataset_ -F _format_first_time_( ex :"%Y-%m-%d %H:%M:%S") -U _time_unit_ -D tXstar tX (see official documentation for more details)



###Launch PeriodiqueGarbreVisualisation_abs.py
Then select the pattern in your output file and paste it as the parameter of the script in a terminal
After you can launch the script.

Time line for nested pattern may need time to load if pattern as a lot of occurrence.



## PeriodiqueGarbreVisualisation_abs_file.py


This is a extention of PeriodiqueGarbreVisualisation_abs.py which allows you to use a list of patterns stored in a file.

File path should be given in parameters.

The patterns should have the same format than for PeriodiqueGarbreVisualisation_abs.py

The file _sacha_18_absI_G60_test-patts.txt_ in this directory contains patterns with the expected format. 

So you can run the script doing: PeriodiqueGarbreVisualisation_abs_file.py sacha_18_absI_G60_text-patts.txt

Same issues are to be expected for time line.



