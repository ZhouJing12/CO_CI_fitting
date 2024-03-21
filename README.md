There are two main python file, CI_P04_expression_test.py and CO_SLED_CI_fit.py.

CI_P04_expression_test.py is used to draw CI population using expression from (Papadopoulos 2004).
run by 
$ python CI_P04_expression_test.py
The plots are style of (Papadopoulos 2021) Figure 5
Using coefficient from (Papadopoulos 2004), the results are the same as (Papadopoulos 2021) Figure 5
In the following analysis, we choose coefficient from LAMDA website.
We also plot iso-density curves for Tcmb=8.62K (Spiderweb) for comparison.
In the following analysis, we choose Tcmb(z)=(1+z) Tcmb(0) for high-z objects.

CO_SLED_CI_fit.py is used to fit CO SLED (LVG) and CI (analytical).
run by
$ python CO_SLED_CI_fit.py
Install package using python-pip if package is required.
If wrapper_my_radex is missing, then run 
$ make
$ make wrapper
in myRadex folder, and move wrapper_my_radex*.so to this folder and replace the older one.

