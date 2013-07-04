package knn;

import java.util.ArrayList;
import java.util.List;

public class KNNSample {

    public static void main(String[] args) {
        int k = 3;

        double[] vectorA = new double[] { 9, 32, 65.1 };
        KNNDTO<String> studentA = new KNNDTO<String>("vectorA", vectorA, "A");

        double[] vectorB = new double[] { 12, 65, 86.1 };
        KNNDTO<String> studentB = new KNNDTO<String>("vectorB", vectorB, "C");

        double[] vectorC = new double[] { 19, 54, 45.1 };
        KNNDTO<String> studentC = new KNNDTO<String>("vectorC", vectorC, "C");

        List<KNNDTO<String>> trainingSet = new ArrayList<KNNDTO<String>>();
        trainingSet.add(studentA);
        trainingSet.add(studentB);
        trainingSet.add(studentC);

        double[] newVector = new double[] { 74, 55, 22 };
        KNNDTO<String> newStudent = new KNNDTO<String>("predictionVector", newVector, "");

        KNNEngine engine = new KNNEngine();
        engine.initialize(newStudent, trainingSet, k);
        engine.EuclideanDistance();

        System.out.println(engine.getResult());

        for (KNNDTO<String> entry : trainingSet) {
            System.out.println("Distance to " + entry.getId() + " : " + entry.getDistance() + " | label: "
                    + entry.getLabel());
        }

    }

}
