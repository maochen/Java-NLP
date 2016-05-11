package org.maochen.nlp.ml.classifier.maxent;

import opennlp.maxent.GISModel;
import opennlp.maxent.io.GISModelReader;
import opennlp.maxent.io.GISModelWriter;
import opennlp.maxent.io.PlainTextGISModelWriter;
import opennlp.model.AbstractModel;
import opennlp.model.DataIndexer;
import opennlp.model.PlainTextFileDataReader;
import opennlp.model.Prior;
import opennlp.model.RealValueFileEventStream;
import opennlp.model.UniformPrior;

import org.maochen.nlp.ml.IClassifier;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.maxent.eventstream.EventStream;
import org.maochen.nlp.ml.classifier.maxent.eventstream.StringEventStream;
import org.maochen.nlp.ml.classifier.maxent.eventstream.TupleEventStream;
import org.maochen.nlp.ml.vector.FeatNamedVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

public class MaxEntClassifier implements IClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(MaxEntClassifier.class);

    private boolean useSmoothing = true;
    private int iterations = 100;
    private int cutoff = 0;
    private int nthreads = Runtime.getRuntime().availableProcessors();
    private double smoothingObservation = 0.1;

    private GISModel model = null;

    public MaxEntClassifier trainString(List<String[]> trainingData) {
        EventStream es = new StringEventStream(trainingData);
        return train(es);
    }

    private MaxEntClassifier train(EventStream es) {
        Prior prior = new UniformPrior();
        DataIndexer di = new OnePassRealValueDataIndexer(es, cutoff, true);

        GISTrainer gisTrainer = new GISTrainer();
        gisTrainer.setSmoothing(useSmoothing);
        gisTrainer.setSmoothingObservation(smoothingObservation);
        model = gisTrainer.trainModel(iterations, di, prior, cutoff, nthreads);
        return this;
    }

    public Map<String, Double> predict(String[] feats) {
        float[] val = RealValueFileEventStream.parseContexts(feats); // This will remove the val in feats
        double[] vector = new double[val.length];
        for (int i = 0; i < val.length; i++) {
            vector[i] = val[i];
        }

        FeatNamedVector featNamedVector = new FeatNamedVector(vector);
        Tuple predict = new Tuple(featNamedVector);

        featNamedVector.featsName = feats;
        return predict(predict);
    }

    @Override
    public IClassifier train(List<Tuple> trainingData) {
        EventStream es = new TupleEventStream(trainingData);
        return train(es);
    }

    @Override
    public Map<String, Double> predict(Tuple predict) {
        if (!(predict.vector instanceof FeatNamedVector)) {
            throw new IllegalArgumentException("Please use FeatNamedVector");
        }

        float[] featureVector = new float[predict.vector.getVector().length];
        for (int i = 0; i < featureVector.length; i++) {
            featureVector[i] = (float) predict.vector.getVector()[i]; // So damn stupid.
        }

        double[] prob = model.eval(((FeatNamedVector) predict.vector).featsName, featureVector, new double[model.getNumOutcomes()]);

        Map<String, Double> resultMap = new HashMap<>();
        for (int i = 0; i < prob.length; i++) {
            resultMap.put(model.getOutcome(i), prob[i]);
        }

        return resultMap;
    }

    @Override
    public void setParameter(Properties paraMap) {
        if (paraMap == null) {
            return;
        }

        if (paraMap.containsKey("use_smoothing")) {
            this.useSmoothing = Boolean.valueOf(paraMap.getProperty("use_smoothing"));
        }

        if (paraMap.containsKey("iter")) {
            this.iterations = Integer.parseInt(paraMap.getProperty("iter"));
        }

        if (paraMap.containsKey("cutoff")) {
            this.cutoff = Integer.parseInt(paraMap.getProperty("cutoff"));
        }

        if (paraMap.containsKey("nthreads")) {
            this.nthreads = Integer.parseInt(paraMap.getProperty("nthreads"));
        }

        if (paraMap.containsKey("smoothing_observation")) {
            this.smoothingObservation = Double.parseDouble(paraMap.getProperty("smoothing_observation"));
        }
    }

    @Override
    public void persistModel(String modelPath) throws IOException {
        File outputFile = new File(modelPath);
        GISModelWriter writer = new PlainTextGISModelWriter(model, outputFile);
        writer.persist();
    }

    @Override
    public void loadModel(InputStream modelPath) {
        LOG.info("Loading MaxEnt model.");
        GISModelReader modelReader = new GISModelReader(new PlainTextFileDataReader(modelPath));
        try {
            AbstractModel model = modelReader.getModel();
            this.model = (GISModel) model;
        } catch (IOException e) {
            LOG.error("model load err.", e);
        }

    }

    public MaxEntClassifier() {
    }
}
