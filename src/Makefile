all:
	g++ -o dense_matrix_test -std=c++11 algebra/TestDenseMatrix.cpp algebra/DenseMatrixT.cpp  -g
	g++ -o file_op_test -std=c++11 utils/TestFileOp.cpp algebra/DenseMatrixT.cpp -g
	g++ -o linear_regress_test -std=c++11 algorithm/regression/TestLinearRegress.cpp algorithm/regression/LinearRegress.cpp algebra/DenseMatrixT.cpp -g
	g++ -o logistic_regression_test -std=c++11 algorithm/regression/TestLogisticRegress.cpp algebra/DenseMatrixT.cpp -g
	g++ -o decision_tree_test -std=c++11 algorithm/tree/TestDecisionTree.cpp algorithm/tree/DecisionTree.cpp algebra/DenseMatrixT.cpp -g
	g++ -o regression_tree_test -std=c++11 algorithm/tree/TestCART.cpp algorithm/tree/CART.cpp algebra/DenseMatrixT.cpp -g
	g++ -o DNN_test -std=c++11 algorithm/nn/TestDNN.cpp algorithm/nn/DNN.cpp algebra/DenseMatrixT.cpp -g
clean:
	rm dense_matrix_test
	rm file_op_test
	rm linear_regress_test
	rm logistic_regression_test
	rm decision_tree_test
	rm regression_tree_test
	rm DNN_test
