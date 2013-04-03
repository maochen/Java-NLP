package knn;

import java.util.ArrayList;
import java.util.List;

public class KNNSample {

    public static void main(String[] args) {
        int k = 3;

        double[] vectorA = new double[] { 9, 32, 65.1 };
        KNNDTO studentA = new KNNDTO(vectorA, "A");

        double[] vectorB = new double[] { 12, 65, 86.1 };
        KNNDTO studentB = new KNNDTO(vectorB, "A");

        double[] vectorC = new double[] { 19, 54, 45.1 };
        KNNDTO studentC = new KNNDTO(vectorC, "C");

        List<KNNDTO> trainingSet = new ArrayList<KNNDTO>();
        trainingSet.add(studentA);
        trainingSet.add(studentB);
        trainingSet.add(studentC);

        double[] newVector = new double[] { 74, 55, 22 };
        KNNDTO newStudent = new KNNDTO(newVector, "");

        KNNEngine engine = new KNNEngine();
        engine.initialize(newStudent, trainingSet, k);
        engine.EuclideanDistance();
        
        System.out.println(engine.getResult());
        
        for(KNNDTO entry:trainingSet){
            System.out.println(entry.getDistance()+" | "+entry.getKey() + " | "+entry.getVector()[0]);
        }

    }

}
