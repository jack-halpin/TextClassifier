
//Sublime Text was here
import java.io.FileNotFoundException;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Random;
import java.util.Scanner;


public class Classifier {
	
	private int NO_OF_DOCUMENTS;		//The number of term vectors listed in the training data
	private int NO_OF_TERMS;			//The length of each term vector
	private int[][] docTermMatrix;		//The 2D array used to store the document-term matrix
	private String[] docLabels;			//A string array used to store the label of each vector in docTermMatrix in parallel
	private String[] uniqueLabels;		//A list of all the unique possible class labels listed in docLabels
	private double[][] similarityMatrix;//The cosine similarity matrix used in evaluation
	
	//Default Constructor
	public Classifier(){
		this.NO_OF_DOCUMENTS = 0;
		this.NO_OF_TERMS = 0;
	}
	
	//This method will find the cosine similarity between two vectors
	public double cosineSimilarity(int[] vectorA, int[] vectorB) {
	    double dotProduct = 0.0;
	    double normVA = 0.0;
	    double normVB = 0.0;
	    for (int i = 0; i < vectorA.length; i++) {
	        dotProduct += vectorA[i] * vectorB[i];
	        normVA += Math.pow(vectorA[i], 2);
	        normVB += Math.pow(vectorB[i], 2);
	    }   
	    
	    return dotProduct / (Math.sqrt(normVA) * Math.sqrt(normVB));
	}
	
	//This function takes in the filenames which contain the document data in Matrix Market Format
	//and the labels for each vector and stores them in the appropriate class variables.
	public void readMatrixData(String matrixFilename, String labelFilename){
		System.out.println("Reading in text data from files...");
		try {
			//First read in the file to create the document term matrix
			FileReader reader = new FileReader(matrixFilename);
			Scanner scn = new Scanner(reader);
			
			//ignore the first line...
			scn.nextLine();
			
			NO_OF_DOCUMENTS = scn.nextInt();
			NO_OF_TERMS = scn.nextInt();

			//Now can discard the next value
			scn.nextLine();
			docTermMatrix = new int[NO_OF_DOCUMENTS][NO_OF_TERMS];
			docLabels = new String[NO_OF_DOCUMENTS];
			
			while(scn.hasNextLine()){
				int a = scn.nextInt();
				int b = scn.nextInt();
				int c = scn.nextInt();
				docTermMatrix[a-1][b-1] = c;
				scn.nextLine();
			}
			
			//Next get the labels for each document in the matrix
			FileReader labelReader = new FileReader(labelFilename);
			Scanner labelscn = new Scanner(labelReader);
			
			while(labelscn.hasNextLine()){
				String[] labelIndex = labelscn.nextLine().split(",");
				docLabels[Integer.parseInt(labelIndex[0]) - 1] = labelIndex[1];
			}	
			
			//It's convenient to have a list of the distinct label types for classification
			uniqueLabels = new HashSet<String>(Arrays.asList(docLabels)).toArray(new String[0]);


		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	//This method creates the similarity matrix used to evaluate the classifier
	public void createSimiliarityMatrix(){
		//This process takes some time depending on the machine running it, best to notify the user of this
		System.out.println("Creating cosine similarity matrix, please wait...");
		
		//Create a new empty array to store the values
		double[][] sim = new double[NO_OF_DOCUMENTS][NO_OF_DOCUMENTS];
		
		//Get the cosine similarities between all the term vectors in the document-term matrix and 
		//store them in the array
		for (int j = 0; j < NO_OF_DOCUMENTS-1; j++){
			int[] row = getRow(docTermMatrix, j);
			sim[j][j] = 1.0;
			for (int i = j+1; i < NO_OF_DOCUMENTS;i++){
				sim[j][i] = cosineSimilarity(row, getRow(docTermMatrix, i));
				sim[i][j] = sim[j][i];
			}
		}
		this.similarityMatrix = sim;
	}
	
	

	
	//Takes in a vector and will classify it based on the data from the rest of the training set
	//NOTE: This function is not used when evaluating the classifier. It is only used when we want
	//to classify a new document that is not in the training set which is not required by the assignment.
	public String predict(int k, int[] v, boolean weighted){
		
		//When a new vector comes need to get the cosine similarity between it and all 
		//other vectors in the training data. Create a 2d array that stores the cosine similarity
		//between the vector and an element, and that element's label for classification
		double[][] cosineSimilarities = new double[NO_OF_DOCUMENTS][2];
		//Loop through the training data and calculate cosine similarities
		for (int i = 0; i < NO_OF_DOCUMENTS; i++){
			cosineSimilarities[i][0] = cosineSimilarity(v, this.docTermMatrix[i]);
			cosineSimilarities[i][1] = i; //need to save the index of the vector in the training set to get it's label later on
		}
		
		return classifyVector(k, cosineSimilarities, weighted);
	}
	
	public String classifyVector(int k, double[][] cosineSimilarities, boolean weighted){
		//First the array of distances has to be sorted. This involves sorting the 2D array based on the value in the first column
		java.util.Arrays.sort(cosineSimilarities, new java.util.Comparator<double[]>() {
		    public int compare(double[] a, double[] b) {
		        if (a[0] > b[0]){
		        	return -1;
		        }
		        else if (a[0] < b[0]){
		        	return 1;
		        }
		        else{
		        	return 0;
		        }
		    }
		});
		
		
		//Now that the most similar documents are known, the classification can begin. First thing to be done
		//is record the score for each possible classification. An array will be used.
		
		//Create an array with the same length as the number of possible classification labels
		double[] scoreKeeper = new double[uniqueLabels.length];
		
		//Iterate through the top k instances
		for (int i = 0; i < k; i++){
			//Get the class label of the sample
			String sampleLabel = this.docLabels[(int)cosineSimilarities[i][1]];
			
			//Find what index in the labels array that this label corresponds to
			for (int j = 0; j < uniqueLabels.length; j++){
				if (sampleLabel.equals(uniqueLabels[j])){
					//If weighted voting is selected, then add the value of the cosine similarity to the vote.
					if (weighted == true){
						scoreKeeper[j] = scoreKeeper[j] + cosineSimilarities[i][0];
					}
					//Otherwise just increment increment the score in that index by 1
					else{
						scoreKeeper[j]++;
					}
				}
			}
		}

		//Next need to get the index of the maximum number in the scoreKeeper array which will correspond
		//to the chosen label from the labels array
		int classLabelIndex = 0;
		for (int i = 1; i < scoreKeeper.length; i++){
//			
			//If the score is greater than change in the index
			if (scoreKeeper[classLabelIndex] < scoreKeeper[i]){
				classLabelIndex = i;
			}
			else if (scoreKeeper[classLabelIndex] == scoreKeeper[i]){
				//If the two scores are the same then choose the max index randomly
				Random r = new Random();
				if (r.nextInt(2) != 0){
					classLabelIndex = i;
				}
				
			}
		}
		return this.uniqueLabels[classLabelIndex];
	}
	
	
	//This function is used in cross validation to classify folds within the training set
	//and return the prediction of each element in the fold
	public String[] evaluateFold(int k, int startIndex, int endIndex, int[] indexArray, boolean weighted){
		//String array to hold the classifications of each vector in the fold
		String[] foldLabels = new String[endIndex - startIndex];
		
		//Value to keep track of which number element in the fold is being examined
		int vectorNumber = 0; 
		
		//Iterate through each vector in the fold and evaluate them
		for (int i = startIndex; i < endIndex; i++){
			
			//Get the current vector we are working with
			int vIndex = indexArray[i];
			
			//Get that vectors distances from the rest of the training data from the similarity matrix
			double[] tempDistances = getRow(this.similarityMatrix, vIndex);
			
			//Now filter out the vectors that we are not including in the classification.
			//This includes the current vector being evaluated and all other vectors in the current fold
			
			double[][] distancesList = new double[(NO_OF_DOCUMENTS - (endIndex - startIndex)) - 1][2];
			
			//Next we fill up this list with the current training set
			int indexTracker = 0;
			for(int j = 0; j < this.NO_OF_DOCUMENTS; j++){
				if (j >= startIndex && j <= endIndex) {
					continue;
				}
				distancesList[indexTracker][0] = (tempDistances[indexArray[j]]);
				distancesList[indexTracker][1] = indexArray[j];
				indexTracker++;
			};
			
			//Classify the vector and add it to the foldLabels array
			String result = classifyVector(k, distancesList, weighted);
			foldLabels[vectorNumber] = result;
			vectorNumber++;
		}
		return foldLabels;
	}
	
	//This function will perform a K fold cross validation on the training set in order to evaluate the model accuracy
	public double kFoldEvaluation(int k, int kN, boolean weighted){
		
		//Rather than computing the cosine distances between each vector everytime we want to evaluate
		//a vector in the training set, the similarity matrix will be used which already contains all the 
		//similarities between each vector
		
		//Variable to hold the final overall classifier accuracy
		double classifierAccuracy = 0;
		
		//If the similiarity matrix for this traning set has not been created then we must create it.
		if (this.similarityMatrix == null){
			createSimiliarityMatrix();
		}

		//Next the rows in the similarity matrix need to be shuffled. Rather than shuffling
		//The data inside the matrix itself, we will shuffle an array of indexes that will be used
		//to refer to the rows in the matrix
		
		//Create array of indexes
		int[] indexArray = new int[this.NO_OF_DOCUMENTS];
		for(int i = 0; i < this.NO_OF_DOCUMENTS; i++){
			indexArray[i] = i;
		}
		
		//Shuffle the indices
		shuffleMatrix(indexArray);
		
		
		//Now perform k fold cross validation. Rather than storing new arrays for the 
		//training and test set each fold, we will just use the indexes, it's much faster
		//and easier to work with
		
		//Initialize the start and stop index for the first fold
		int startIndex = 0;
		int endIndex;
		
		for(int i = 0; i < k; i++){			
			endIndex = startIndex + ((this.NO_OF_DOCUMENTS) / k) - 1;
			
			int samplesInFold = endIndex - startIndex;
			
			//If the number of documents cannot be evenly divided by the number of folds
			//then we have to ensure that the last fold contains the remaining samples
			if((NO_OF_DOCUMENTS-endIndex) < samplesInFold){
				endIndex = this.NO_OF_DOCUMENTS-1;
			}
			
			//Classify each vector in the current fold
			String[] foldResults = evaluateFold(kN, startIndex, endIndex, indexArray, weighted);
			//Get the accuracy the predictions
			int score = 0;
			int classIndex = 0;
			
			for(int j = startIndex; j < endIndex; j++){
				if (foldResults[classIndex].equals(this.docLabels[indexArray[j]])){
					score++;
				}
				classIndex++;
			}
			
			
			double eval = ((double)score/(double)samplesInFold) * 100;
			classifierAccuracy = classifierAccuracy + eval;
			startIndex = endIndex + 1;
		}
		
		//Get the average accuracy
		classifierAccuracy = classifierAccuracy/k;
		//Return the accuracy
		return classifierAccuracy;
	}
	
	
	
	//This method will return a single row in a 2D integer array
	public int[] getRow(int[][] m, int row){
		int row_size = m[0].length;
		int[] a = new int[row_size];
		
		for (int i = 0; i < row_size; i++){
			a[i] = m[row][i];
		}
		return a;
	}
	
	//This method will return a single row in a 2D double array
	public double[] getRow(double[][] m, int row){
		int row_size = m[0].length;
		double[] a = new double[row_size];
		
		for (int i = 0; i < row_size; i++){
			a[i] = m[row][i];
		}
		return a;
	}
	
	
	//This method is used to shuffle values in an integer array
	//Used for shuffling the indexes during evaluation
	public int[] shuffleMatrix(int[] m){
		Random rnd = new Random();
		for (int i = m.length - 1; i > 0; i--)
		{
			int index = rnd.nextInt(i + 1);
		    // Simple swap
		    int a = m[index];
		    m[index] = m[i];
		    m[i] = a;
		}
		return m;
	}
	
	//Method to return the size of the training set
	public int getNoOfSamples(){
		return this.NO_OF_DOCUMENTS;
	}
	
}
