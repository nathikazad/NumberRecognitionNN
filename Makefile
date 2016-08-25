all:main.cpp
	g++ -O2 -g -o nn main.cpp mnist-utils.cpp NeuralNetwork.cpp Layer.cpp Neuron.cpp  -I.
ICC = icc
IFLAGS = -O2 -openmp -g -restrict -debug inline-debug-info

cpu:main.cpp
	$(ICC) $(IFLAGS) -o nn_cpu main.cpp mnist-utils.cpp NeuralNetwork.cpp Layer.cpp Neuron.cpp  -I.

phi:main.cpp
	$(ICC) $(IFLAGS) -mmic -o nn_phi main.cpp mnist-utils.cpp NeuralNetwork.cpp Layer.cpp Neuron.cpp  -I.
