package org.maochen.nlp.classifier.maxent;

import opennlp.maxent.GISModel;
import opennlp.maxent.io.GISModelReader;
import opennlp.maxent.io.GISModelWriter;
import opennlp.maxent.io.PlainTextGISModelWriter;
import opennlp.model.AbstractModel;
import opennlp.model.DataIndexer;
import opennlp.model.Prior;
import opennlp.model.RealValueFileEventStream;
import opennlp.model.UniformPrior;

import org.maochen.nlp.classifier.maxent.eventstream.EventStream;
import org.maochen.nlp.classifier.maxent.eventstream.StringEventStream;
import org.maochen.nlp.classifier.maxent.eventstream.TupleEventStream;
import org.maochen.nlp.datastructure.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MaxEntClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(MaxEntClassifier.class);

    private boolean USE_SMOOTHING = true;
    private static final int ITERATIONS = 100;
    private static final int CUTOFF = 0;
    private static final int THREADS = 1;// Runtime.getRuntime().availableProcessors();
    private static final double SMOOTHING_OBSERVATION = 0.1;

    private GISModel model = null;
    private String pathPrefix = MaxEntClassifier.class.getResource(".").getPath();

    public MaxEntClassifier trainString(List<String[]> trainingData) {
        EventStream es = new StringEventStream(trainingData);
        return train(es);
    }

    public MaxEntClassifier train(List<Tuple> trainingData) {
        EventStream es = new TupleEventStream(trainingData);
        return train(es);
    }

    private MaxEntClassifier train(EventStream es) {
        Prior prior = new UniformPrior();
        DataIndexer di = new OnePassRealValueDataIndexer(es, CUTOFF, true);

        GISTrainer gisTrainer = new GISTrainer();
        gisTrainer.setSmoothing(USE_SMOOTHING);
        gisTrainer.setSmoothingObservation(SMOOTHING_OBSERVATION);
        model = gisTrainer.trainModel(ITERATIONS, di, prior, CUTOFF, THREADS);
        return this;
    }

    public Map<String, Double> predict(String[] predictStr) {
        Tuple predict = new Tuple(null);
        float[] val = RealValueFileEventStream.parseContexts(predictStr);
        predict.featureVector = new double[val.length];
        for (int i = 0; i < val.length; i++) {
            predict.featureVector[i] = val[i];
        }
        predict.featureName = predictStr;
        return predict(predict);
    }

    public Map<String, Double> predict(Tuple predict) {
        float[] featureVector = new float[predict.featureVector.length];
        for (int i = 0; i < featureVector.length; i++) {
            featureVector[i] = (float) predict.featureVector[i]; // So damn stupid.
        }

        double[] prob = model.eval(predict.featureName, featureVector, new double[model.getNumOutcomes()]);

        Map<String, Double> resultMap = new HashMap<>();
        for (int i = 0; i < prob.length; i++) {
            resultMap.put(model.getOutcome(i), prob[i]);
        }

        return resultMap;
    }

    public void persist(String modelPath) throws IOException {
        File outputFile = new File(modelPath);
        GISModelWriter writer = new PlainTextGISModelWriter(model, outputFile);
        writer.persist();
    }

    public void loadModel(String modelPath) throws IOException {
        LOG.info("Loading MaxEnt model. ");
        GISModelReader modelReader = new GISModelReader(new File(modelPath));
        AbstractModel model = modelReader.getModel();
        this.model = (GISModel) model;
    }

    public MaxEntClassifier(boolean useSmoothing) {
        this.USE_SMOOTHING = useSmoothing;
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
        maxent.trainString(traindata);

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

//        [home, pdiff=0.6875, ptwins=0.5]
//        {tie=0.1899, lose=0.3686, win=0.4416}
//        [home, pdiff=1.0625, ptwins=0.5]
//        {tie=0.2462, lose=0.3451, win=0.4087}
//        [away, pdiff=0.8125, ptwins=0.5]
//        {tie=0.2251, lose=0.5947, win=0.1802}
//        [away, pdiff=0.6875, ptwins=0.6]
//        {tie=0.1882, lose=0.6056, win=0.2062}
//        [home, pdiff=0.9375, ptwins=0.5]
//        {tie=0.2263, lose=0.3535, win=0.4202}
//        [home, pdiff=0.6875, ptwins=0.3333]
//        {tie=0.2305, lose=0.3859, win=0.3836}
//        [away, pdiff=1.0625, ptwins=0.6666]
//        {tie=0.2294, lose=0.5656, win=0.205}
//        [home, pdiff=0.8125, ptwins=0.6666]
//        {tie=0.1688, lose=0.3409, win=0.4903}
//        [home, pdiff=0.9375, ptwins=0.3333]
//        {tie=0.272, lose=0.3665, win=0.3614}
//        [home, pdiff=0.6875, ptwins=0.5]
//        {tie=0.1899, lose=0.3686, win=0.4416}
        for (String[] predict : predictData) {
            System.out.println(Arrays.toString(predict));
            System.out.println(maxent.predict(predict));
        }
    }
}
