package MLP;

public class Utils {

	public static final int DATASET_SIZE = 8000;
	public static final int K = 4;
	public static final int D = 2;
	public static final double LEARNING_RATE = 0.005;
	public static final double ERROR_THRESHOLD = 0.00001; 

    public enum ActivationFunction {
        NONE {
            public double activate(double x) {
                return x; 
            }
            public double derivative(double x) {
                return 1; 
            }
        },

        TANH {
            public double activate(double x) {
                return Math.tanh(x);
            }
            public double derivative(double x) {
                return 1 - Math.pow(Math.tanh(x), 2);
            }
        },

        RELU {
            public double activate(double x) {
                return Math.max(0, x); 
            }
            public double derivative(double x) {
                return x > 0 ? 1 : 0;  
            }
        },

        SIGMOID {
            public double activate(double x) {
                return 1 / (1 + Math.exp(-x)); 
            }
            public double derivative(double x) {
                double sigmoid = activate(x);
                return sigmoid * (1 - sigmoid);  
            }
        };

		public abstract double activate(double x);
        public abstract double derivative(double x);
	}		
}
