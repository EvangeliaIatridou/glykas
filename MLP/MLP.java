package MLP;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import MLP.Utils.ActivationFunction;

class MLP{

	private double errorThreshold = Utils.ERROR_THRESHOLD;

	private int outputSize;
	private Layer[] layers;
	private int layersNum;
	private double[] finalOutput;
	
	public MLP(int outputSize, int[] layerSizes, ActivationFunction[] activationFunctions){
		this.outputSize = outputSize;
		layersNum = layerSizes.length;
		layers = new Layer[layersNum];
		boolean isInputLayer;
		
		for (int i = 0; i < layersNum; i++) {
			isInputLayer = i == 0;
			layers[i] = new Layer(layerSizes[i], (i == 0 ? 1 : layerSizes [i -1]), activationFunctions[i], isInputLayer);
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
			counter++;
		}
		return interDerivatives;
		
	}

	public void forwardPass(double[] initialInput) {
		layers[0].setNeuronInputs(initialInput);
	
		for (int i = 0; i < layersNum - 1; i++) { 
			double[] outputs = layers[i].computeOutputs();
			layers[i + 1].setNeuronInputs(outputs);
		}
	
		finalOutput = layers[layersNum - 1].computeOutputs();
	}
	
	public void backwardPass(int[] target) {
		double[] outputDeltas = new double[outputSize];
		double[] layerDerivatives = layers[layersNum - 1].computeDerivatives();
		double error = 0;

		for (int i = 0; i < outputSize; i++) {
			error = finalOutput[i] - target[i];
			outputDeltas[i] = error * layerDerivatives[i];
		}
		layers[layersNum - 1].updateDeltas(outputDeltas);

		for (int i = layersNum - 2; i > 0; i--) {
			Layer currentLayer = layers[i];
			Layer nextLayer = layers[i+1];
					
			double[] currentDeltas = currentLayer.computeDeltas(nextLayer.getWeights(), outputDeltas);
			outputDeltas = currentDeltas;
		}

		updatePartialDerivatives();
	}

	public void initializePartialDerivatives() {
		for (int i = 0; i < layersNum; i ++) {
			layers[i].setPartialDerivatives(0);
		}
	}

	public void updatePartialDerivatives() {
		for (int i = 0; i < layersNum; i ++) {
			layers[i].updatePartialDerivatives();
		}
	}

	public void updateWeights() {
		for (int i = 1; i < layersNum; i ++) {
			layers[i].updateWeights();
		}
	}
	
	public void gradientDescentWithBatches(double[][] inputs, int[][] targets, int batchSize) {
		int batches = inputs.length / batchSize;
		int epoch = 0;
		double errorDifference = Double.MAX_VALUE;
		double previousError = 0;
		double currentError = 0;

		while (true) {
			initializePartialDerivatives();
			currentError = 0;

			for (int i = 0; i < batches; i++) {
				for (int j = i*batchSize; j < ((i+1)*batchSize); j++) {
					forwardPass(inputs[j]);
					backwardPass(targets[j]);
					currentError += calculateError(targets[j]);
				}

				updateWeights();
			}

			currentError /= inputs.length;
			System.out.println("Epoch " + epoch + ": E(w)= " + currentError);

			if (epoch > 0) {
				errorDifference = Math.abs(currentError - previousError);
			}

			previousError = currentError;
			epoch++;
			
			if (epoch > 800) {
				if ((errorDifference < errorThreshold) || epoch > 4000) {
					System.out.println("\nTraining terminated at epoch " + (epoch-1) + " with error difference " + errorDifference);
        			return;
				}
			}
		}
		
	}

	private double calculateError(int[] target) {
		double error = 0.0;
		double difference = 0.0;

		for (int i = 0; i < outputSize; i ++) {
			difference = finalOutput[i] - target[i];
			error += difference * difference;
		}

		return error / outputSize;
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
		String outData = "";
		boolean predictedCorrectly = false;
		
		for (int i = 0; i < testInputs.length; i++) {

			predictedCorrectly = false;
			forwardPass(testInputs[i]);
			int predictedCategory = getPredictedCategory();

			if (testLabels[i][predictedCategory] == 1) {
				predictedCorrectly = true;
				correctPredictions++;
			}

			for(int j = 0; j<testInputs[i].length;j++){
                outData += testInputs[i][j] + ","; //somehow i have to find the actual labels and turn the one who isnt predicted correctly into 0 
            }

			if (predictedCorrectly) outData += (predictedCategory+1) + "\n";
			else outData += "0" + "\n";
		}

		testResult = (correctPredictions / (double) testLabels.length)*100;
		System.out.println("Test Accuracy: " + testResult + "% !!!");

        String filePath = "predictions.txt";

        try(BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))){
            writer.write(outData);
        }catch(IOException e){
            System.err.println("error occured: unable to write MLP data.");
            System.exit(0);
        }

	}
}
