package MLP;
import java.util.Random;

import MLP.Utils.ActivationFunction;

class Neuron{

	private double learningRate = Utils.LEARNING_RATE;

	private double[] inputs;
	private double[] weights;
	private double[] partialDerivatives;
	private double totalInput;
	private double output;
	private double delta;
	private boolean isInputNeuron;
	
	private ActivationFunction activationFunction;

	public Neuron(int numInputs, ActivationFunction activationFunction, boolean isInputNeuron) {
		this.activationFunction = activationFunction;
		this.isInputNeuron = isInputNeuron;

        weights = new double[numInputs+1];
		partialDerivatives = new double[numInputs+1];
        inputs = new double[numInputs]; 
		output = 0;

		Random r = new Random();
	
		for (int i = 0; i < numInputs+1; i++) {
			weights[i] = (2 * r.nextDouble()) - 1; // random double between -1 and 1

			if (isInputNeuron) {
				weights[0] = 0;
				weights[1] = 1;
			}
        }
	}

	public void computeTotalInput(){
		if (isInputNeuron) {
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
		output = activationFunction.activate(totalInput);
		return output;
	}

	public double computeDerivative() {
		return activationFunction.derivative(output);
	}
	
	public double updateInterDerivative(double input, double interDerivative){ 
		
		interDerivative += delta*input; 

		return interDerivative;
	}

	public void updateWeights() {
		for (int i = 0; i < weights.length; i++) {
			weights[i] -= learningRate * partialDerivatives[i];
		}
	}

	public void updatePartialDerivatives() {
		partialDerivatives[0] = delta;

		for (int i = 1; i < partialDerivatives.length; i++) {
			partialDerivatives[i] = delta * inputs[i-1];
		}
	}

	public void setPartialDerivatives(double value) {
		for (int i = 1; i < partialDerivatives.length; i++)
		partialDerivatives[i] = value;	
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