package classifier.maxent;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import classifier.maxent.gis.*;

public class MaxEntClassifier implements IClassifier {
    private boolean USE_SMOOTHING = false;
    private double SMOOTHING_OBSERVATION = 0.1;
    private int CUTOFF = 0;
    private int ITERATIONS = 100;

    GISModel model = null;

    Map<String, Double> resultMap = null;

    @Override
    public IClassifier train(List<String[]> trainingdata) {
        DataIndexer di = new DataIndexer(trainingdata, CUTOFF);
        model = new GISTrainer(true, USE_SMOOTHING, SMOOTHING_OBSERVATION).trainModel(ITERATIONS, di, CUTOFF);
        return this;
    }

    // Predict only single feature vector.
    @Override
    public Map<String, Double> predict(String[] featureVector) {
        if (model == null) throw new RuntimeException("Model is null.");
        double[] ocs;
        float[] realVals = new float[featureVector.length];

        // Find Reals
        boolean hasReal = false;
        for (int i = 0; i < featureVector.length; i++) {
            int loc = featureVector[i].lastIndexOf("=") + 1;
            if (loc == 0 || loc == featureVector[i].length()) realVals[i] = 1;
            else {
                hasReal = true;
                realVals[i] = Float.parseFloat(featureVector[i].substring(loc));
                featureVector[i] = featureVector[i].substring(0, loc - 1);
                if (realVals[i] < 0) {
                    throw new RuntimeException("Negitive values are not allowed: " + realVals[i]);
                }
            }
        }
        if (!hasReal) realVals = null;
        // ------------

        ocs = model.eval(featureVector, realVals);

        // <Group Tag, Probability>
        Map<String, Double> result = new HashMap<String, Double>();
        for (int i = 0; i < model.getNumOutcomes(); i++) {
            result.put(model.getLabelFromIndex(i), ocs[i]);
        }

        resultMap = result;
        return resultMap;
    }

    @Override
    public String getResult() {
        if (resultMap == null) throw new RuntimeException("Predicting First");

        double max = 0;
        String maxTag = null;

        for (String key : resultMap.keySet()) {
            if (resultMap.get(key) > max) {
                max = resultMap.get(key);
                maxTag = key;
            }
        }

        return maxTag;
    }

    public void persist(String loc) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(loc))) {
            out.writeObject(model);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void readModel(String loc) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(loc));) {
            model = (GISModel) in.readObject();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public MaxEntClassifier(boolean usesmoothing, double smoothingObv) {
        this.USE_SMOOTHING = usesmoothing;
        this.SMOOTHING_OBSERVATION = smoothingObv;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        List<String[]> traindata = new ArrayList<String[]>();
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5", "win" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5", "win" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6666", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.3333", "win" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.6666", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.3333", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.75", "win" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.25", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.25", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6", "tie" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.25", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.25", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.25", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.6", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.4", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6666", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.4", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.7142", "win" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5", "win" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5714", "win" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.625", "win" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.4285", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5714", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.5555", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5555", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.5", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6", "win" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5555", "win" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.6", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.5454", "win" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5", "win" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.6", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.4444", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.4545", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5454", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.5384", "tie" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.4545", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5454", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5454", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.5384", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5833", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.5714", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5384", "win" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5384", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.5384", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6", "tie" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5714", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5333", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.4666", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.625", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5333", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.4375", "win" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6470", "win" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5333", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5294", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.4117", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6111", "tie" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5625", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5294", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.4444", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6111", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5882", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5555", "win" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.4736", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6315", "win" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5882", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5263", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.4736", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5882", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.55", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.45", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6190", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5882", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.55", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.4285", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6363", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5882", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5714", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.4545", "lose" });

        MaxEntClassifier maxent = new MaxEntClassifier(false, 0);
        maxent.train(traindata);

        maxent.persist("fixture/maxentModel.txt");
        maxent.model = null;
        maxent.readModel("fixture/maxentModel.txt");

        List<String[]> predictData = new ArrayList<String[]>();
        predictData.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.5" });
        predictData.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5" });
        predictData.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5" });
        predictData.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6" });
        predictData.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5" });
        predictData.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.3333" });
        predictData.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.6666" });
        predictData.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.6666" });
        predictData.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.3333" });
        predictData.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.5" });

        for (String[] feature : predictData) {
            System.out.println(Arrays.toString(feature));
            System.out.println(maxent.predict(feature));
        }

    }
}
