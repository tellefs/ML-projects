## Project 2

Collaboration between Tellef Storebakken (@tellefs), Julian E. Vevik (@jvevik) and Maria L. Markova (@sagora123 / @markmarko).

The final report, as well as the .tex file and figures used can be found in the Latex folder. The final report is called project2_report.pdf. 

The Files folder contain textfiles with data for the different methods and runs. 

The Figures folder contain the Figures produces.

The src folder holds all the source .py files we created and used to find the results in this project. They are used in the Task_*.py files.

An example run is included in the interactive project2.ipynb notebook. Here all the Task programs were run with the settings described below. This notebook also contains the Scikit-Learn and TensorFlow/Keras NN for classification. If you encounter problems in opening the notebook in Github.com you can easily view it from the following link: https://nbviewer.jupyter.org/github/tellefs/ML-projects/blob/master/project2/project2.ipynb


Programs Task_a.py, Task_b.py, Task_c.py, Task_d.py and Task_e.py were used for collecting the results for the project. These programs are run in the project2.ipynb jupyter notebook as an example run.
With the current settings chosen in the programs:

	1. task_a_py runs a simple analysis of the training and test MSE for ridge with lambda=0.001, stochastic gradient descent with 1000 epochs and constant eta=0.001.

	2. task_b.py runs the self-made FFNN with three hidden layers (50 neurons each) with 1000 epochs, eta=0.001 and lambda=0.001 (ridge case)

	3. task_c.py runs the self-made NN with one hidden layer (50 neurons) with 1000 epochs, eta=0.001 and tanh activation function for the hidden layers.

	4. task_d.py runs the self-made FFNN with three hidden layers (50 neurons each) with 100 epochs, eta=0.01 and lambda=0.001 (ridge case) for the classification problem.

	5. task_e.py runs logistic regresion with stochastic gradient descent with 100 epochs and eta=0.01.

	Keras NN for the classification case is placed in project_2.ipynb
