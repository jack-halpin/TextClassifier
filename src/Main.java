

import java.util.Random;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

public class Main {
	
	public static void main(String[] args) {
		//Create a new classifier object that is used for kNN Text classification
		Classifier kNN = new Classifier();
		
		//Read in the data from the two data files and store them to the class variables
		//This could be considered training the class
		kNN.readMatrixData("text_data/news_articles.mtx", "text_data/news_articles.labels");
		
		//Perform evaluation on the classifier based on the training set.
		//10 fold cross validation from K = 1 to 10 without weighted kNN
		System.out.println("Beginning non-Weighted Evaluation");
		System.out.println("==========================");
		double nonsum = 0;
		long  start = System.currentTimeMillis();
		for (int k = 1; k <= 10; k++){
			double j = kNN.kFoldEvaluation(10,k, false);
			nonsum = nonsum + j;
			System.out.println("Non Weighted classifier with k = " + k + " has " + j + "% accuracy.");
			
		}
		
		//Perform evaluation on the classifier based on the training set.
		//10 fold cross validation from K = 1 to 10 with weighted kNN
		System.out.println("");
		System.out.println("Beginning weighted Evaluation");
		System.out.println("==========================");
		double weightedsum = 0;
		for (int k = 1; k <= 10; k++){
			double j = kNN.kFoldEvaluation(10, k, true);
			weightedsum = weightedsum + j;
			System.out.println("Weighted classifier with k = " + k + " has " + j + "% accuracy.");
		}
		System.out.println("Non-weighted average = " + nonsum/(double)10);
		System.out.println("Weighted average = " + weightedsum/(double)10);
		long finish = System.currentTimeMillis();
		
		System.out.print(finish-start);
		
		
		
	}

}
