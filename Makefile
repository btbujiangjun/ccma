CC=g++
all:
	${CC} -o dense_matrix_test -std=c++11 examples/algebra/TestDenseMatrix.cpp src/algebra/BaseMatrix.cpp src/algebra/DenseMatrix.cpp  -g -pthread -I ./include/
	${CC} -o file_op_test -std=c++11 examples/utils/TestFileOp.cpp src/algebra/BaseMatrix.cpp src/algebra/DenseMatrix.cpp -pthread -g -I ./include/
	${CC} -o MnistHelper_test -std=c++11 examples/utils/TestMnistHelper.cpp src/algebra/DenseMatrix.cpp -pthread -g -I ./include/
	${CC} -o linear_regress_test -std=c++11 examples/algorithm/regression/TestLinearRegress.cpp src/algorithm/regression/LinearRegress.cpp src/algebra/BaseMatrix.cpp src/algebra/DenseMatrix.cpp -pthread -g -I ./include/
	${CC} -o logistic_regression_test -std=c++11 examples/algorithm/regression/TestLogisticRegress.cpp src/algebra/BaseMatrix.cpp src/algebra/DenseMatrix.cpp -pthread -g -I ./include/
	${CC} -o decision_tree_test -std=c++11 examples/algorithm/tree/TestDecisionTree.cpp src/algorithm/tree/DecisionTree.cpp src/algebra/BaseMatrix.cpp src/algebra/DenseMatrix.cpp -pthread -g -I ./include/
	${CC} -o regression_tree_test -std=c++11 examples/algorithm/tree/TestCART.cpp src/algorithm/tree/CART.cpp src/algebra/BaseMatrix.cpp src/algebra/DenseMatrix.cpp -pthread -g -I ./include/
	${CC} -o DNN_test -std=c++11 examples/algorithm/nn/TestDNN.cpp src/algorithm/nn/DNN.cpp src/algebra/BaseMatrix.cpp src/algebra/DenseMatrix.cpp src/algorithm/nn/Cost.cpp -g -pthread -Wall -O3 -I ./include/
clean:
	rm -rf dense_matrix_test* &
	rm -rf file_op_test* &
	rm -rf MnistHelper_test* &
	rm -rf linear_regress_test* &
	rm -rf logistic_regression_test* &
	rm -rf decision_tree_test* &
	rm -rf regression_tree_test* &
	rm -rf DNN_test* &
