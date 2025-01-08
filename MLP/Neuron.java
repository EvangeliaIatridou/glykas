package MLP;
import java.util.Random;

import MLP.Utils.ActivationFunction;

class Neuron{

	private double[] inputs;
	private double[] weights;
	private double totalInput;
	private double output;
	private double delta;
	
	private ActivationFunction activationFunction;

	public Neuron(int numInputs, ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;

        weights = new double[numInputs+1];
        inputs = new double[numInputs]; 
		output = 0;

		Random r = new Random();
	
		for (int i = 0; i < numInputs+1; i++) {
			weights[i] = (2 * r.nextDouble()) - 1; // random double between -1 and 1

			if (activationFunction == ActivationFunction.NONE) {
				weights[0] = 0;
				weights[1] = 1;
			}
        }
	}

	public void computeTotalInput(){
		if (activationFunction == ActivationFunction.NONE) {
			totalInput = inputs[0];
			return;
		}

		double sum = weights[0]; //bias
		
		for(int i = 1; i < weights.length; i++){
			sum += weights[i] * inputs[i-1]; 
		}

		totalInput = sum;
	}
	
	public double computeOutput() {
		computeTotalInput();
		switch (activationFunction) {
			case TANH:
				output = Math.tanh(totalInput);
				break;
			case RELU:
				output = (totalInput > 0) ? totalInput : 0; //relu
				break;
			case SIGMOID:
				output = 1.0 / (1.0 + Math.exp(-totalInput)); // sigmoid
				break;
			default:
				output = totalInput;
		}
		return output;
	}

	public double computeDerivative() {
		double derivative;
	
		switch (activationFunction) {
			case TANH:
				derivative = 1 - Math.pow(output, 2);
				break;
			case RELU:
				derivative = (output > 0) ? 1 : 0;
				break;
			case SIGMOID:
				derivative = output * (1 - output);
				break;
			default:
				derivative = 1; 
		}
		return derivative;
	}
	
	public double updateInterDerivative(double input, double interDerivative){ 
		
		interDerivative += delta*input; 

		return interDerivative;
	}

	public void updateWeights(double learningRate, double interDerivative) {
		weights[0] -= learningRate * delta;
		for (int i = 1; i < weights.length; i++) {
			weights[i] -= learningRate * interDerivative;
		}
	}
	
	public void setInputs(double[] inputs){
		this.inputs = inputs;
	}

	public double[] getInputs(){
		return inputs;
	}
	
	public void setWeights(double[] weights){
		this.weights = weights;
	}
	
	public double[] getWeights(){
		return weights;
	}

	public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getDelta() {
        return delta;
    }

}