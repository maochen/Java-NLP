package knn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Simple Wrapper, Id is based on the input sequential.
 * 
 * @author MaochenG
 * 
 */
public class KNN {

    private List<KNNDTO<String>> trainSetDTOList;
    private KNNDTO<String> predictDTO;

    private int idCounter;

    private KNNEngine engine;

    private KNNDTO<String> inputConverter(String[] entry, boolean isPredict) {
        int newLength = (isPredict) ? entry.length : entry.length - 1;
        double[] vector = new double[newLength];
        String label = null;

        for (int i = 0; i < entry.length; i++) {
            if (!isPredict && i == (entry.length - 1)) {
                label = entry[i];
                break;
            }

            vector[i] = Double.parseDouble(entry[i]);
        }
        KNNDTO<String> entryDTO = new KNNDTO<String>(idCounter++, vector, label);
        return entryDTO;
    }

    // mode: 0 - EuclideanDistance, 1 - ChebyshevDistance, 2 - ManhattanDistance
    public String predict(String[] predict, int k, int mode) {
        predictDTO = inputConverter(predict, true);
        engine.initialize(predictDTO, trainSetDTOList, k);

        if (mode == 1) {
            engine.ChebyshevDistance();
        }
        else if (mode == 2) {
            engine.ManhattanDistance();
        }
        else {
            engine.EuclideanDistance();
        }

        return engine.getResult();
    }

    /**
     * Return the predict to every other train vector's distance.
     * 
     * @return return by Id which is ordered by input sequential.
     */
    public Map<Integer, Double> getDetails() {
        Map<Integer, Double> outputMap = new HashMap<Integer, Double>();

        for (KNNDTO<String> dtos : trainSetDTOList) {
            outputMap.put(dtos.getId(), dtos.getDistance());
        }
        return outputMap;
    }

    public KNN(List<String[]> trainSet) {
        this.idCounter = 0;
        engine = new KNNEngine();
        this.trainSetDTOList = new ArrayList<KNNDTO<String>>();

        for (String[] entry : trainSet) {
            this.trainSetDTOList.add(inputConverter(entry, false));
        }

    }
}
