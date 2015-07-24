package org.maochen.nlp.maochen.nlp.classifier.maxent;

import opennlp.maxent.GIS;
import opennlp.maxent.GISModel;
import opennlp.maxent.io.GISModelWriter;
import opennlp.maxent.io.SuffixSensitiveGISModelReader;
import opennlp.maxent.io.SuffixSensitiveGISModelWriter;
import opennlp.model.AbstractModel;
import opennlp.model.EventStream;
import opennlp.model.RealValueFileEventStream;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class MaxEntClassifier {

    private boolean USE_SMOOTHING = true;
    private static final int ITERATIONS = 100;
    private static final int CUTOFF = 0;

    GISModel model = null;

    Map<String, Double> resultMap = null;

    String pathPrefix = MaxEntClassifier.class.getResource(".").getPath();

    private MaxEntClassifier train(List<String[]> trainingData) {
        try {
            EventStream es = new StringEventStream(trainingData);
            //            EventStream es = new TupleEventStream(trainingData);
            model = GIS.trainModel(es, ITERATIONS, CUTOFF, USE_SMOOTHING, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return this;
    }

    public Map<String, Double> predict(String[] featureVector) {

        // TODO: here we dont need to worry about feature labels but we need to change the above train!
        //        String[] contexts = new String[featureVector.length];
        //
        //        for (int i = 0; i < featureVector.length; i++) {
        //            contexts[i] = String.valueOf(i);
        //        }

        float[] values = RealValueFileEventStream.parseContexts(featureVector);
        double[] ocs = model.eval(featureVector, values);
        String outcomes = model.getAllOutcomes(ocs);

        resultMap = new HashMap<>();
        String[] outcomeEntries = outcomes.split("\\s+");
        for (String outcomeEntry : outcomeEntries) {
            String key = outcomeEntry.split("\\[")[0];
            double value = Double.parseDouble(outcomeEntry.split("\\[")[1].split("\\]")[0]);
            resultMap.put(key, value);
        }
        return resultMap;
    }

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

    public void setParameter(Map<String, String> paraMap) {

    }

    public void persist(String modelPath) throws IOException {
        File outputFile = new File(modelPath);
        GISModelWriter writer = new SuffixSensitiveGISModelWriter(model, outputFile);
        writer.persist();
    }

    public void loadModel(String modelPath) throws IOException {
        long start = System.currentTimeMillis();
        System.out.println("Loading MaxEnt model ... ");
        AbstractModel model = new SuffixSensitiveGISModelReader(new File(modelPath)).getModel();
        long end = System.currentTimeMillis();
        long duration = (end - start) / 1000;
        System.out.println("completed ... " + duration + " secs.");

        this.model = (GISModel) model;
    }

    public MaxEntClassifier(boolean usesmoothing) {
        this.USE_SMOOTHING = usesmoothing;
    }

    public static void main(String[] args) throws IOException {
        List<String[]> traindata = new ArrayList<>();
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5", "win"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5", "win"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6666", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.3333", "win"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.6666", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.3333", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.75", "win"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.25", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.25", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6", "tie"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.25", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.25", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.25", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.6", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.4", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6666", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.4", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.7142", "win"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5", "win"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5714", "win"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.625", "win"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.4285", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5714", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.5555", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5555", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.5", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6", "win"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5555", "win"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.6", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.5454", "win"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5", "win"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.6", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.4444", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.4545", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5454", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.5384", "tie"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.4545", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5454", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5454", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.5384", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5833", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.5714", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5384", "win"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5384", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.5384", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6", "tie"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5714", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5333", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.4666", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.625", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5333", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.4375", "win"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6470", "win"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5333", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5294", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.4117", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6111", "tie"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5625", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5294", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.4444", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6111", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5882", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5555", "win"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.4736", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6315", "win"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5882", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5263", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.4736", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5882", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.55", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.45", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6190", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5882", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.55", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.4285", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6363", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5882", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5714", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.4545", "lose"});


        MaxEntClassifier maxent = new MaxEntClassifier(false);
        maxent.train(traindata);

        maxent.persist(maxent.pathPrefix + "/maxentModel.txt");
        maxent.model = null;
        maxent.loadModel(maxent.pathPrefix + "/maxentModel.txt");

        List<String[]> predictData = new ArrayList<String[]>();
        predictData.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.5"});
        predictData.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5"});
        predictData.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5"});
        predictData.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6"});
        predictData.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5"});
        predictData.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.3333"});
        predictData.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.6666"});
        predictData.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.6666"});
        predictData.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.3333"});
        predictData.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.5"});

        /*
        * [home, pdiff=0.6875, ptwins=0.5]
        * {lose=0.328, tie=0.2409, win=0.4311}
        * [home, pdiff=1.0625, ptwins=0.5]
        * {lose=0.3415, tie=0.2284, win=0.4301}
        * [away, pdiff=0.8125, ptwins=0.5]
        * {lose=0.5586, tie=0.2552, win=0.1862}
        * [away, pdiff=0.6875, ptwins=0.6]
        * {lose=0.5574, tie=0.2562, win=0.1863}
        * [home, pdiff=0.9375, ptwins=0.5]
        * {lose=0.337, tie=0.2325, win=0.4305}
        * [home, pdiff=0.6875, ptwins=0.3333]
        * {lose=0.322, tie=0.2466, win=0.4314}
        * [away, pdiff=1.0625, ptwins=0.6666]
        * {lose=0.5782, tie=0.238, win=0.1838}
        * [home, pdiff=0.8125, ptwins=0.6666]
        * {lose=0.3385, tie=0.2311, win=0.4304}
        * [home, pdiff=0.9375, ptwins=0.3333]
        * {lose=0.331, tie=0.2381, win=0.4309}
        * [home, pdiff=0.6875, ptwins=0.5]
        * {lose=0.328, tie=0.2409, win=0.4311}
            */
        for (String[] feature : predictData) {
            System.out.println(Arrays.toString(feature));
            System.out.println(maxent.predict(feature));
        }
    }
}
