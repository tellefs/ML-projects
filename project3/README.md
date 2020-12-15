## Project 3

Collaboration between Tellef Storebakken (@tellefs), Julian E. Vevik (@jvevik) and Maria L. Markova (@sagora123 / @markmarko)

The final report, as well as the .tex file and figures used in the report can be found in the Latex folder. The final report is called **project3_report.pdf**.

The Files folder contain textfiles with data for the different methods and runs, and the data can be plotted using the **print_and_plot.py** program found in the src folder.

The Figures folder contain the Figures produced.

The src folder holds all the source **.py** files we created and used to find the results in this project. They are used in the different analysis files discussed below.

An example run is included in the interactive **project3_testrun.ipynb** notebook. Here all the analysis programs were run with the settings described below. If you encounter problems in opening the notebook in Github.com you can easily view it from the following link: https://nbviewer.jupyter.org/github/tellefs/ML-projects/blob/master/project3/project3_testrun.ipynb.


The programs **linear_regression_analysis.py**, **neural_network_analysis.py**, **decisiontree_analysis.py**, **XGB_analysis.py**, **article_analysis.py**, **nn_linear_article_analysis.py**, **decisiontree_article_analysis.py** and **XGB_article_analysis.py** were used for collecting the results for the project. These programs are run in the **project3_testrun.ipynb** jupyter notebook as an example run, with the current settings chosen in the programs:

	1. Blank

	2. task_b.py runs the self-made FFNN with three hidden layers (50 neurons each) with 1000 epochs, eta=0.001 and lambda=0.001 (ridge case)

  3. decisiontree_analysis.py runs the decision tree method from SciKit-Learn using a max depth of 14 and lambda from 10^{-1} to 10^{-7}.
