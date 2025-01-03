package MLP;
import java.util.Random;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math;


class DataSDT{

	private double[][] inputs;
	private int[][] labels;
	private int dataset_size;
	private String filePath;

	public DataSDT() {
		dataset_size = Utils.DATASET_SIZE;
		filePath = "dataSDT.txt";
		inputs = new double[dataset_size][Utils.D];
		labels = new int[dataset_size][Utils.K];
	}

	public void generateData() {
		Random rnd = new Random();
		
		int category = 0;
		
		double[] x1 = new double[dataset_size];
		double[] x2 = new double[dataset_size];
		
		String data = "";
		
		for(int i=0; i<dataset_size; i++){ 
			
			//generating random (x1,x2) in [-1,1]x[-1,1]:		
			
			x1[i] = -1+rnd.nextDouble()*2;
			x2[i] = -1+rnd.nextDouble()*2;
			
			//assigning teams C1 through C4:
			
						
			if((Math.pow((x1[i]-0.5),2)+Math.pow((x2[i]-0.5),2)<0.2 && x2[i]>0.5) || 
			(Math.pow((x1[i]+0.5),2)+Math.pow((x2[i]+0.5),2)<0.2 && x2[i]>-0.5) || 
			(Math.pow((x1[i]-0.5),2)+Math.pow((x2[i]+0.5),2)<0.2 && x2[i]>-0.5) || 
			(Math.pow((x1[i]+0.5),2)+Math.pow((x2[i]-0.5),2)<0.2 && x2[i]>0.5)){ //for C1
				
				category = 1;
			
			}else if((Math.pow((x1[i]-0.5),2)+Math.pow((x2[i]-0.5),2)<0.2 && x2[i]<0.5) || 
			(Math.pow((x1[i]+0.5),2)+Math.pow((x2[i]+0.5),2)<0.2 && x2[i]<-0.5) || 
			(Math.pow((x1[i]-0.5),2)+Math.pow((x2[i]+0.5),2)<0.2 && x2[i]<-0.5) || 
			(Math.pow((x1[i]+0.5),2)+Math.pow((x2[i]-0.5),2)<0.2 && x2[i]<0.5)){ //for C2

				category = 2;

			}else{

				if(x1[i]*x2[i]>0){ //for C3
					category = 3;

				}else{ //for C4
					category = 4;
				}
			}			
			
			data += x1[i]+","+x2[i]+","+category+"\n";
			
		}
		
		try(BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))){
			writer.write(data);
		}catch(IOException e){
			System.err.println("error occured: unable to write SDT data.");
			System.exit(0);
		}

		System.out.println("File dataSDT.txt generated successfully!");
		
	}

	public void loadData() {
		
		BufferedReader reader =null;
		try {
			reader = new BufferedReader(new FileReader(filePath));
		}
		catch(FileNotFoundException e) {
			System.out.println("File was not found");
			System.out.println("or could not be opened.");
			System.exit(0);
		}
		try {
			String line = reader.readLine();
			int counter=0;
			while (line != null) {
				line = line.trim();
				String[] split = line.split(",");
				
				double x1 = Double.parseDouble(split[0].trim());
				double x2 = Double.parseDouble(split[1].trim());
				int t = Integer.parseInt(split[2].trim());

				inputs[counter][0] = x1;
				inputs[counter][1] = x2;
				labels[counter][t-1] = 1; 

				line = reader.readLine(); // read next line
				if(counter<dataset_size) // prints data in cmd
				{
					/*System.out.println(counter + ": " + inputs[counter][0]
						+ ", " + inputs[counter][1] + ", t: (" 
						+ labels[counter][0] + ", " + labels[counter][1] + ", "
						+ labels[counter][2] + ", " + labels[counter][3] + ")");*/
					counter++;
				}
			}
			reader.close();
		}
		catch(IOException e) {
			System.out.println("IOException");
			System.out.println("or could not be opened.");
			System.exit(0);
		}
	}
	
	public double[][] getInputs() { 
		return inputs;
	}

	public int[][] getLabels() { 
		return labels;
	}
}