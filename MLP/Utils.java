package MLP;

public class Utils {

	public static final int DATASET_SIZE = 8000;
	public static final int K = 4;
	public static final int D = 2;
	public static final double LEARNING_RATE = 0.0001;
	public static final double ERROR_THRESHOLD = 0.00001; 

    public enum ActivationFunction {
		NONE,
		TANH,
		RELU,
		SIGMOID;
	}	
}
