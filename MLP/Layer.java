package MLP;

import MLP.Utils.ActivationFunction;

class Layer {

	private int size;
	private Neuron[] neurons;
	private ActivationFunction activationFunction;

    public Layer(int size, int previousLayerSize, ActivationFunction activationFunction) {
        this.size = size;
        this.neurons = new Neuron[size];
		this.activationFunction = activationFunction;

        for (int i = 0; i < size; i++) {
            neurons[i] = new Neuron(previousLayerSize, activationFunction);
        }
    }

	public void setNeuronInputs(double[] inputs) {
		for (int i = 0; i < size; i ++) {
			switch (activationFunction) {

				case NONE:
					double[] initialInput = new double[1];
					initialInput[0] = inputs[i];
					neurons[i].setInputs(initialInput);
					break;

				default:
					neurons[i].setInputs(inputs);
			}
		}
	}

	public double[] computeOutputs() {
		double[] outputs = new double[size];

		for (int i = 0; i < size; i++) {
			outputs[i] = neurons[i].computeOutput();
		}

		return outputs;
	}

	public double[] computeDerivatives() {
		double[] derivatives = new double[size];

		for (int i = 0; i < size; i++) {
			derivatives[i] = neurons[i].computeDerivative();
		}

		return derivatives;
	}

	public double[] computeDeltas(double[][] nextLayerWeights, double[] nextLayerDeltas) {
		double[] deltas = new double[size];

		for (int i = 0; i < size; i++) {
			double newDelta = 0;
			for (int j = 0; j < nextLayerDeltas.length; j++) {
				newDelta += nextLayerDeltas[j] * nextLayerWeights[j][i]; 
			}

			newDelta *= neurons[i].computeDerivative();
			neurons[i].setDelta(newDelta);
			deltas[i] = newDelta;
		}

		return deltas;
	}

	public void updateWeights(double learningRate, double interDerivative) {
        for (Neuron neuron : neurons) {
            neuron.updateWeights(learningRate,interDerivative);
        }
    }

	public double updateInterDerivative(double interDerivative) { 
        for (Neuron neuron : neurons) {
			for(int i =0;i<neuron.getInputs().length;i++){
				interDerivative = neuron.updateInterDerivative(neuron.getInputs()[i],interDerivative);
			}
		}
		return interDerivative;
    }

	public void updateDeltas(double[] deltas) {
		for (int i = 0; i < size; i ++) {
			neurons[i].setDelta(deltas[i]);
		}
	}

	public double[][] getWeights() {
		double[][] weights = new double[size][];
		for (int i = 0; i < size; i++) {
			weights[i] = neurons[i].getWeights();
		}
		return weights;
	}
	
	public void setNeurons(Neuron[] neurons){
		this.neurons = neurons;
	}
	
	public Neuron[] getNeurons(){
		return neurons;
	}
	
	public int getSize() {
		return size;
	}

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}
}