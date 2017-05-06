all:
	g++ -o dense_matrix_test -std=c++11 algebra/TestDenseMatrix.cpp algebra/BaseMatrix.cpp algebra/DenseMatrix.cpp  -g -pthread -I ./include/
	g++ -o file_op_test -std=c++11 utils/TestFileOp.cpp algebra/BaseMatrix.cpp algebra/DenseMatrix.cpp -pthread -g -I ./include/
	g++ -o MnistHelper_test -std=c++11 utils/TestMnistHelper.cpp ./algebra/DenseMatrix.cpp -pthread -g -I ./include/
	g++ -o linear_regress_test -std=c++11 algorithm/regression/TestLinearRegress.cpp algorithm/regression/LinearRegress.cpp algebra/BaseMatrix.cpp algebra/DenseMatrix.cpp -pthread -g -I ./include/
	g++ -o logistic_regression_test -std=c++11 algorithm/regression/TestLogisticRegress.cpp algebra/BaseMatrix.cpp algebra/DenseMatrix.cpp -pthread -g -I ./include/
	g++ -o decision_tree_test -std=c++11 algorithm/tree/TestDecisionTree.cpp algorithm/tree/DecisionTree.cpp algebra/BaseMatrix.cpp algebra/DenseMatrix.cpp -pthread -g -I ./include/
	g++ -o regression_tree_test -std=c++11 algorithm/tree/TestCART.cpp algorithm/tree/CART.cpp algebra/BaseMatrix.cpp algebra/DenseMatrix.cpp -pthread -g -I ./include/
	g++ -o DNN_test -std=c++11 algorithm/nn/TestDNN.cpp algorithm/nn/DNN.cpp algebra/BaseMatrix.cpp algebra/DenseMatrix.cpp -g -pthread -Wall -O3 -I ./include/
clean:
	rm -rf dense_matrix_test* &
	rm -rf file_op_test* &
	rm -rf MnistHelper_test* &
	rm -rf linear_regress_test* &
	rm -rf logistic_regression_test* &
	rm -rf decision_tree_test* &
	rm -rf regression_tree_test* &
	rm -rf DNN_test* &
