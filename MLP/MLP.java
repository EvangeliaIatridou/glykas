package MLP;

import MLP.Utils.ActivationFunction;

class MLP{

	//private int inputSize;
	private int outputSize;
	private Layer[] layers;
	private int layersNum;
	private double[] finalOutput;
	
	public MLP(int outputSize, int[] layerSizes, ActivationFunction[] activationFunctions){
		//this.inputSize = inputSize;
		this.outputSize = outputSize;
		layersNum = layerSizes.length;
		layers = new Layer[layersNum];
		
		for (int i = 0; i < layersNum; i++) {
			layers[i] = new Layer(layerSizes[i], (i == 0 ? 1 : layerSizes [i -1]), activationFunctions[i]);
		}
	}

	public void displayMLPArchitecture() {
        System.out.println("\n------MLP Architecture------\n");
        for (int i = 0; i < layersNum; i++) {
            System.out.println("Layer: " + i + ", Neurons: " + layers[i].getSize() + ", Function: " + layers[i].getActivationFunction());
        }
		System.out.println("\n----------------------------\n");
    }

	public double[] updateInterDerivative(double interDerivative){
		double[] interDerivatives = new double[layers.length];
		int counter=0;
		for(Layer layer: layers){
			interDerivatives[counter] = layer.updateInterDerivative(interDerivative);
			//System.out.println("ssssss "+interDerivatives[counter]);
			counter++;
		}
		return interDerivatives;
		
	}

	public void forwardpass(double[] initialInput) {
		layers[0].setNeuronInputs(initialInput);
	
		for (int i = 0; i < layersNum - 1; i++) { 
			double[] outputs = layers[i].computeOutputs();
			layers[i + 1].setNeuronInputs(outputs);
		}
	
		finalOutput = layers[layersNum - 1].computeOutputs();
	}
	
	public void backwardpass(int[] target, double learningRate) {
		double[] outputDeltas = new double[outputSize];
		Layer outputLayer = layers[layersNum - 1];
		double[] layerDerivatives = outputLayer.computeDerivatives();

		for (int i = 0; i < outputSize; i++) {
			double error = finalOutput[i] - target[i];
			outputDeltas[i] = error * layerDerivatives[i];
		}
		outputLayer.updateDeltas(outputDeltas);

		for (int i = layersNum - 2; i > 0; i--) {
			Layer currentLayer = layers[i];
			Layer nextLayer = layers[i+1];
					
			double[] currentDeltas = currentLayer.computeDeltas(nextLayer.getWeights(), outputDeltas);
			outputDeltas = currentDeltas;
		}

		/*
		for (Layer layer : layers) {  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			layer.updateWeights(learningRate);
		}*/
	}

	public double computeError(int[][] targets) {
		double totalError = 0;
		for (int i = 0; i < targets.length; i++) {
            double error = 0;
            for (int j = 0; j < outputSize; j++) {
                error += Math.pow(targets[i][j] - finalOutput[j], 2);
            }
            totalError += error;
        }
        return (totalError / targets.length)*0.5;
		//return totalError / targets.length;
	}
	
	public void gradientDescentWithBatches(double[][] inputs, int[][] targets, double learningRate, int batchSize, double errorThreshold) {
		int batches = inputs.length / batchSize;
		int epoch = 0;
		double errorDifference = Double.MAX_VALUE;
		double previousError = 0;
		//double interDerivative; //added

		while (true) {

			for (int i = 0; i < batches; i++) {
				for (int j = i*batchSize; j < ((i+1)*batchSize); j++) {
					forwardpass(inputs[j]);
					backwardpass(targets[j], learningRate);
				}
				double interDerivative = 0;
				double[] interDerivatives;
				interDerivatives = updateInterDerivative(interDerivative);

				//System.out.println("iiiiiiiiii "+interDerivative); //debug
				//update weights
				int counter = 0;
				for (Layer layer : layers) { 
					
					layer.updateWeights(learningRate,interDerivatives[counter]); //gonna expand with interDerivative
					counter++;
				}
			}

			double currentError = computeError(targets);
			if (epoch > 0) {
                errorDifference = Math.abs(currentError - previousError);
            }
			previousError = currentError;
			System.out.println("Epoch " + epoch + " Error: " + currentError);

			epoch++;
			
			if (epoch > 800) {
				if (errorDifference < errorThreshold) {
					System.out.println("\nTraining terminated at epoch " + (epoch-1) + " with error difference " + errorDifference);
        			return;
				}
			}
		}
		
	}

	private int getPredictedCategory() {
        int category = 0;
        for (int i = 1; i < finalOutput.length; i++) {
            if (finalOutput[i] > finalOutput[category]) {
                category = i;
            }
        }
        return category;
    }

	public void test(double[][] testInputs, int[][] testLabels) {
		int correctPredictions = 0;
		double testResult = 0.0;
		
		for (int i = 0; i < testInputs.length; i++) {
			forwardpass(testInputs[i]);
			int predictedCategory = getPredictedCategory();
			if (testLabels[i][predictedCategory] == 1) {
				correctPredictions++;
			}
		}
		testResult = (correctPredictions / (double) testLabels.length)*100;
		System.out.println("Test Accuracy: " + testResult + "% !!!");
	}
}
