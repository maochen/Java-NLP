package classifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import classifier.knn.KNNDTO;
import classifier.knn.KNNEngine;

/**
 * Simple Wrapper, Id is based on the input sequential.
 * 
 * @author MaochenG
 * 
 */
public class KNNClassifier implements IClassifier {

    private List<KNNDTO<String>> trainSetDTOList;
    private KNNDTO<String> predictDTO;

    private int idCounter;
    private int k;
    private int mode;

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
    
    @Override
    // mode: 0 - EuclideanDistance, 1 - ChebyshevDistance, 2 - ManhattanDistance
    public void setParameter(Map<String, String> paraMap) {
        if (paraMap.containsKey("k")) this.k = Integer.parseInt(paraMap.get("k"));
        if (paraMap.containsKey("mode")) this.mode = Integer.parseInt(paraMap.get("mode"));
    }

    public KNNClassifier() {
        this.idCounter = 0;
        this.k = 1;
        this.mode = 0;
        engine = new KNNEngine();
    }

    /**
     * train() method for knn is just used for loading trainingdata!!
     */
    @Override
    public IClassifier train(List<String[]> trainingdata) {
        this.trainSetDTOList = new ArrayList<KNNDTO<String>>();

        for (String[] entry : trainingdata) {
            this.trainSetDTOList.add(inputConverter(entry, false));
        }

        return this;
    }

    /**
     * Return the predict to every other train vector's distance.
     * 
     * @return return by Id which is ordered by input sequential.
     */
    @Override
    public Map<String, Double> predict(String[] predict) {
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

        engine.getResult();

        Map<String, Double> outputMap = new HashMap<String, Double>();

        for (KNNDTO<String> dtos : trainSetDTOList) {
            outputMap.put(String.valueOf(dtos.getId()), dtos.getDistance());
        }
        return outputMap;
    }

    @Override
    public String getResult() {
        return predictDTO.getLabel();
    }

}
