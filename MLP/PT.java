package MLP;

import java.util.Scanner;
import MLP.Utils.ActivationFunction;


//auth tha einai h klassh me ta main kai pou tha epilegei o xrhsths an thelei 2 h 3 layers of neurons gia na ylopoihsei antistoixa to PT2 h to PT3
class PT {

	private int k = Utils.K;
	private int inputSize = Utils.D;
	private int dataset_size = Utils.DATASET_SIZE;
	private double learning_rate = Utils.LEARNING_RATE;
	private double error_threshold = Utils.ERROR_THRESHOLD;

	private int layers; 
	private int[] layerSizes;
	private int batchSize;
	private ActivationFunction[] activationFunctions;

	public void defineArchitecture() {
		layers = 0;
		batchSize = 0;

		Scanner in = new Scanner(System.in);

		//System.out.print("Define the number of input neurons (d): ");
		//int d = in.nextInt();


		System.out.print("\nDefine the number of hidden layers (pick 2 or 3): "); 
		layers = in.nextInt();
		while (layers != 2 && layers != 3) {
			System.out.print("Invalid choice. Choose between 2 and 3: ");
			layers = in.nextInt();
		} 

		
		layerSizes = new int[layers + 2];
		layerSizes[0] = inputSize;
		layerSizes[layers+1] = k;

		activationFunctions = new ActivationFunction[layers + 2];
		activationFunctions[0] = ActivationFunction.NONE;
		activationFunctions[layers + 1] = ActivationFunction.SIGMOID;

		for (int i = 1; i <= layers; i++) {

			System.out.print("\nDefine the number of neurons in hidden layer " + i +": ");
        	layerSizes[i] = in.nextInt();
			
			System.out.print("Choose activation function for hidden layer " + i + " (1: tanh or 2: ReLU): ");
			int funtionChoice = in.nextInt();

			while (funtionChoice != 1 && funtionChoice != 2) {
				System.out.print("Invalid choice. Choose between 1 and 2: ");
				funtionChoice = in.nextInt();
			}

			if (funtionChoice == 1) activationFunctions[i] = ActivationFunction.TANH;
			else activationFunctions[i] = ActivationFunction.RELU;
		}


		System.out.print("\nDefine the mini-batch size (B) for training: N/");
    	batchSize = dataset_size / in.nextInt();

		in.close();
	}

	public void displayArchitecture() {
		System.out.println("\n--------Architecture--------\n");
		System.out.println("Number of hidden layers: " + layers 
			+ "\nNumber of input neurons (d): " + layerSizes[0]
			+ "\nNumber of categories (K): " + layerSizes[layers+1]
			+ "\nHidden layer neurons (H1 H2" + (layers == 3? " H3): " : "): ")
			+ layerSizes[1] +" "+ layerSizes[2] 
			+" "+ (layers == 3? layerSizes[3] : "")
			+ "\nActivation Function for each hidden layer: " + activationFunctions[1] +" "+ activationFunctions[2] +" "
			+ (layers == 3? activationFunctions[3] : "")
			+ "\nMini-batch size (B): N/" + (dataset_size / batchSize));
			System.out.println("\n----------------------------\n");
	}

	public static void main(String[] args){
		PT pt = new PT();
		pt.defineArchitecture();
		pt.displayArchitecture();

		DataSDT dataSDT = new DataSDT();
		dataSDT.generateData();
		dataSDT.loadData();
		double[][] inputs = dataSDT.getInputs();
		int[][] targets = dataSDT.getLabels();

		int halfDatasetSize = Utils.DATASET_SIZE / 2;

		double[][] trainingInputs = new double[halfDatasetSize][Utils.D];
		int[][] trainingTargets = new int[halfDatasetSize][Utils.K];
		System.arraycopy(inputs, 0, trainingInputs, 0, halfDatasetSize);
		System.arraycopy(targets, 0, trainingTargets, 0, halfDatasetSize);

		double[][] controlInputs = new double[halfDatasetSize][Utils.D];
		int[][] controlTargets = new int[halfDatasetSize][Utils.K];
		System.arraycopy(inputs, halfDatasetSize, controlInputs, 0, halfDatasetSize);
		System.arraycopy(targets, halfDatasetSize, controlTargets, 0, halfDatasetSize);
 

		MLP mlp = new MLP(pt.k, pt.layerSizes, pt.activationFunctions);
		mlp.displayMLPArchitecture();

		mlp.gradientDescentWithBatches(trainingInputs, trainingTargets, pt.learning_rate, pt.batchSize, pt.error_threshold);

		double testResult = mlp.test(controlInputs, controlTargets);
		System.out.println("Test Accuracy: " + testResult + "%");

		//to compute ikanothta genikeushs we just do testresult/100
	}
}
