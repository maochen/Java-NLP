package classifier.maxent.gis;

import classifier.maxent.gis.GISModel.EvalParameters;
import classifier.maxent.gis.GISModel.Context;

/**
 * An implementation of Generalized Iterative Scaling. The reference paper for this implementation
 * was Adwait Ratnaparkhi's tech report at the University of Pennsylvania's Institute for Research
 * in Cognitive Science, and is available at <a href
 * ="ftp://ftp.cis.upenn.edu/pub/ircs/tr/97-08.ps.Z">
 * <code>ftp://ftp.cis.upenn.edu/pub/ircs/tr/97-08.ps.Z</code></a>.
 * 
 * The slack parameter used in the above implementation has been removed by default from the
 * computation and a method for updating with Gaussian smoothing has been added per Investigating
 * GIS and Smoothing for Maximum Entropy Taggers, Clark and Curran (2002). <a
 * href="http://acl.ldc.upenn.edu/E/E03/E03-1071.pdf">
 * <code>http://acl.ldc.upenn.edu/E/E03/E03-1071.pdf</code></a> The slack parameter can be used by
 * setting <code>useSlackParameter</code> to true. Gaussian smoothing can be used by setting
 * <code>useGaussianSmoothing</code> to true.
 * 
 * A prior can be used to train models which converge to the distribution which minimizes the
 * relative entropy between the distribution specified by the empirical constraints of the training
 * data and the specified prior. By default, the uniform distribution is used as the prior.
 * 
 * @author Tom Morton
 * @author Jason Baldridge
 * @version $Revision: 1.7 $, $Date: 2010/09/06 08:02:18 $
 */
public final class GISTrainer {

    /**
     * Specifies whether unseen context/outcome pairs should be estimated as occur very
     * infrequently.
     */
    private boolean useSimpleSmoothing = false;
    /**
     * Specifies whether a slack parameter should be used in the model.
     */
    private boolean useSlackParameter = false;
    /**
     * Specified whether parameter updates should prefer a distribution of parameters which is
     * gaussian.
     */
    private boolean useGaussianSmoothing = false;
    private double sigma = 2.0;

    // If we are using smoothing, this is used as the "number" of
    // times we want the trainer to imagine that it saw a feature that it
    // actually didn't see. Defaulted to 0.1.
    private double _smoothingObservation = 0.1;

    private boolean printMessages = false;

    /** Number of unique events which occurred in the event set. */
    private int numUniqueEvents;
    /** Number of predicates. */
    private int numPreds;
    /** Number of outcomes. */
    private int numOutcomes;

    /** Records the array of predicates seen in each event. */
    private int[][] contexts;

    /** The value associated with each context. If null then context values are assumes to be 1. */
    private float[][] values;

    /** List of outcomes for each event i, in context[i]. */
    private int[] outcomeList;

    /** Records the num of times an event has been seen for each event i, in context[i]. */
    private int[] numTimesEventsSeen;

    /** The number of times a predicate occured in the training data. */
    private int[] predicateCounts;

    private int cutoff;

    /**
     * Stores the String names of the outcomes. The GIS only tracks outcomes as ints, and so this
     * array is needed to save the model to disk and thereby allow users to know what the outcome
     * was in human understandable terms.
     */
    private String[] outcomeLabels;

    /**
     * Stores the String names of the predicates. The GIS only tracks predicates as ints, and so
     * this array is needed to save the model to disk and thereby allow users to know what the
     * outcome was in human understandable terms.
     */
    private String[] predLabels;

    /** Stores the observed expected values of the features based on training data. */
    private Context[] observedExpects;

    /** Stores the estimated parameter value of each predicate during iteration */
    private Context[] params;

    /** Stores the expected values of the features based on the current models */
    private Context[] modelExpects;

    /** This is the prior distribution that the model uses for training. */
    // private UniformPrior prior;

    /** Observed expectation of correction feature. */
    private double cfObservedExpect;
    /** A global variable for the models expected value of the correction feature. */
    private double CFMOD;

    private final double LLThreshold = 0.0001;

    /**
     * Stores the output of the current model on a single event durring training. This we be reset
     * for every event for every iteration.
     */
    private double[] modelDistribution;

    /** Initial probability for all outcomes. */
    private EvalParameters evalParams;

    /**
     * Train a model using the GIS algorithm.
     * 
     * @param iterations The number of GIS iterations to perform.
     * @param di The data indexer used to compress events in memory.
     * @param modelPrior The prior distribution used to train this model.
     * @return The newly trained model, which can be used immediately or saved to disk using an
     *         opennlp.maxent.io.GISModelWriter object.
     */

    // 143 -148
    public GISModel trainModel(int iterations, DataIndexer di, int cutoff) {

        /************** Incorporate all of the needed info ******************/
        display("Incorporating indexed data for training...  \n");
        contexts = di.contexts;
        values = di.values;
        this.cutoff = cutoff;
        predicateCounts = di.predCounts;
        numTimesEventsSeen = di.numTimesEventsSeen;
        numUniqueEvents = contexts.length;
        // printTable(contexts);

        outcomeLabels = di.outcomeLabels;
        numOutcomes = outcomeLabels.length;

        outcomeList = di.outcomeList;

        predLabels = di.predLabels;

        numPreds = predLabels.length;

        display("\tNumber of Event Tokens: " + numUniqueEvents + "\n");
        display("\t    Number of Outcomes: " + numOutcomes + "\n");
        display("\t  Number of Predicates: " + numPreds + "\n");

        // determine the correction constant and its inverse
        int correctionConstant = 1;
        for (int ci = 0; ci < contexts.length; ci++) {
            if (values == null || values[ci] == null) {
                if (contexts[ci].length > correctionConstant) {
                    correctionConstant = contexts[ci].length;
                }
            }
            else {
                float cl = values[ci][0];
                for (int vi = 1; vi < values[ci].length; vi++) {
                    cl += values[ci][vi];
                }

                if (cl > correctionConstant) {
                    correctionConstant = (int) Math.ceil(cl);
                }
            }
        }
        display("done.\n");

        // set up feature arrays
        float[][] predCount = new float[numPreds][numOutcomes];
        for (int ti = 0; ti < numUniqueEvents; ti++) {
            for (int j = 0; j < contexts[ti].length; j++) {
                if (values != null && values[ti] != null) {
                    predCount[contexts[ti][j]][outcomeList[ti]] += numTimesEventsSeen[ti] * values[ti][j];
                }
                else {
                    predCount[contexts[ti][j]][outcomeList[ti]] += numTimesEventsSeen[ti];
                }
            }
        }

        // A fake "observation" to cover features which are not detected in
        // the data. The default is to assume that we observed "1/10th" of a
        // feature during training.
        final double smoothingObservation = _smoothingObservation;

        // Get the observed expectations of the features. Strictly speaking,
        // we should divide the counts by the number of Tokens, but because of
        // the way the model's expectations are approximated in the
        // implementation, this is cancelled out when we compute the next
        // iteration of a parameter, making the extra divisions wasteful.
        params = new Context[numPreds];
        modelExpects = new Context[numPreds];
        observedExpects = new Context[numPreds];

        // The model does need the correction constant and the correction feature. The correction
        // constant
        // is only needed during training, and the correction feature is not necessary.
        // For compatibility reasons the model contains form now on a correction constant of 1,
        // and a correction param 0.
        evalParams = new EvalParameters(params, 0, 1, numOutcomes);
        int[] activeOutcomes = new int[numOutcomes];
        int[] outcomePattern;
        int[] allOutcomesPattern = new int[numOutcomes];
        for (int oi = 0; oi < numOutcomes; oi++) {
            allOutcomesPattern[oi] = oi;
        }
        int numActiveOutcomes = 0;
        for (int pi = 0; pi < numPreds; pi++) {
            numActiveOutcomes = 0;
            if (useSimpleSmoothing) {
                numActiveOutcomes = numOutcomes;
                outcomePattern = allOutcomesPattern;
            }
            else { // determine active outcomes
                for (int oi = 0; oi < numOutcomes; oi++) {
                    if (predCount[pi][oi] > 0 && predicateCounts[pi] >= cutoff) {
                        activeOutcomes[numActiveOutcomes] = oi;
                        numActiveOutcomes++;
                    }
                }
                if (numActiveOutcomes == numOutcomes) {
                    outcomePattern = allOutcomesPattern;
                }
                else {
                    outcomePattern = new int[numActiveOutcomes];
                    for (int aoi = 0; aoi < numActiveOutcomes; aoi++) {
                        outcomePattern[aoi] = activeOutcomes[aoi];
                    }
                }
            }
            params[pi] = new Context(outcomePattern, new double[numActiveOutcomes]);
            modelExpects[pi] = new Context(outcomePattern, new double[numActiveOutcomes]);
            observedExpects[pi] = new Context(outcomePattern, new double[numActiveOutcomes]);
            for (int aoi = 0; aoi < numActiveOutcomes; aoi++) {
                int oi = outcomePattern[aoi];
                params[pi].parameters[aoi] = 0.0;
                modelExpects[pi].parameters[aoi] = 0.0;
                if (predCount[pi][oi] > 0) {
                    observedExpects[pi].parameters[aoi] = predCount[pi][oi];
                }
                else if (useSimpleSmoothing) {
                    observedExpects[pi].parameters[aoi] = smoothingObservation;
                }
            }
        }

        // compute the expected value of correction
        if (useSlackParameter) {
            int cfvalSum = 0;
            for (int ti = 0; ti < numUniqueEvents; ti++) {
                for (int j = 0; j < contexts[ti].length; j++) {
                    int pi = contexts[ti][j];
                    if (!modelExpects[pi].contains(outcomeList[ti])) {
                        cfvalSum += numTimesEventsSeen[ti];
                    }
                }
                cfvalSum += (correctionConstant - contexts[ti].length) * numTimesEventsSeen[ti];
            }
            if (cfvalSum == 0) {
                cfObservedExpect = Math.log(0.01); // nearly zero so log is defined
            }
            else {
                cfObservedExpect = Math.log(cfvalSum);
            }
        }
        predCount = null; // don't need it anymore

        display("...done.\n");

        modelDistribution = new double[numOutcomes];
        // numfeats = new int[numOutcomes];

        /***************** Find the parameters ************************/
        display("Computing model parameters...\n");
        findParameters(iterations, correctionConstant);

        /*************** Create and return the model ******************/
        // To be compatible with old models the correction constant is always 1
        return new GISModel(params, predLabels, outcomeLabels, 1, evalParams.getCorrectionParam(), new UniformPrior());

    }

    /* Estimate and return the model parameters. */
    private void findParameters(int iterations, int correctionConstant) {
        double prevLL = 0.0;
        double currLL = 0.0;
        display("Performing " + iterations + " iterations.\n");

        UniformPrior prior = new UniformPrior();
        prior.setLabels(outcomeLabels, predLabels);

        for (int i = 1; i <= iterations; i++) {
            if (i < 10) display("  " + i + ":  ");
            else if (i < 100) display(" " + i + ":  ");
            else display(i + ":  ");
            currLL = nextIteration(correctionConstant, prior);
            if (i > 1) {
                if (prevLL > currLL) {
                    System.err.println("Model Diverging: loglikelihood decreased");
                    break;
                }
                if (currLL - prevLL < LLThreshold) {
                    break;
                }
            }
            prevLL = currLL;
        }

        // kill a bunch of these big objects now that we don't need them
        observedExpects = null;
        modelExpects = null;
        numTimesEventsSeen = null;
        contexts = null;
    }

    // modeled on implementation in Zhang Le's maxent kit
    private double gaussianUpdate(int predicate, int oid, int n, double correctionConstant) {
        double param = params[predicate].parameters[oid];
        double x = 0.0;
        double x0 = 0.0;
        double f;
        double tmp;
        double fp;
        double modelValue = modelExpects[predicate].parameters[oid];
        double observedValue = observedExpects[predicate].parameters[oid];
        for (int i = 0; i < 50; i++) {
            tmp = modelValue * Math.exp(correctionConstant * x0);
            f = tmp + (param + x0) / sigma - observedValue;
            fp = tmp * correctionConstant + 1 / sigma;
            if (fp == 0) {
                break;
            }
            x = x0 - f / fp;
            if (Math.abs(x - x0) < 0.000001) {
                x0 = x;
                break;
            }
            x0 = x;
        }
        return x0;
    }

    /* Compute one iteration of GIS and return log-likelihood. */
    private double nextIteration(int correctionConstant, UniformPrior prior) {
        // compute contribution of p(a|b_i) for each feature and the new
        // correction parameter
        double loglikelihood = 0.0;
        CFMOD = 0.0;
        int numEvents = 0;
        int numCorrect = 0;
        for (int ei = 0; ei < numUniqueEvents; ei++) {

            prior.logPrior(modelDistribution, contexts[ei]);
            GISModel.eval(contexts[ei], values[ei], modelDistribution, evalParams);

            for (int j = 0; j < contexts[ei].length; j++) {
                int pi = contexts[ei][j];
                if (predicateCounts[pi] >= cutoff) {
                    int[] activeOutcomes = modelExpects[pi].outcomes;
                    for (int aoi = 0; aoi < activeOutcomes.length; aoi++) {
                        int oi = activeOutcomes[aoi];
                        if (values != null && values[ei] != null) {
                            modelExpects[pi].parameters[aoi] += modelDistribution[oi] * values[ei][j]
                                    * numTimesEventsSeen[ei];
                        }
                        else {
                            modelExpects[pi].parameters[aoi] += modelDistribution[oi] * numTimesEventsSeen[ei];
                        }
                    }
                    if (useSlackParameter) {
                        for (int oi = 0; oi < numOutcomes; oi++) {
                            if (!modelExpects[pi].contains(oi)) {
                                CFMOD += modelDistribution[oi] * numTimesEventsSeen[ei];
                            }
                        }
                    }
                }
            }
            if (useSlackParameter) CFMOD += (correctionConstant - contexts[ei].length) * numTimesEventsSeen[ei];

            loglikelihood += Math.log(modelDistribution[outcomeList[ei]]) * numTimesEventsSeen[ei];
            numEvents += numTimesEventsSeen[ei];
            if (printMessages) {
                int max = 0;
                for (int oi = 1; oi < numOutcomes; oi++) {
                    if (modelDistribution[oi] > modelDistribution[max]) {
                        max = oi;
                    }
                }
                if (max == outcomeList[ei]) {
                    numCorrect += numTimesEventsSeen[ei];
                }
            }

        }
        display(".");

        // compute the new parameter values
        for (int pi = 0; pi < numPreds; pi++) {
            double[] observed = observedExpects[pi].parameters;
            double[] model = modelExpects[pi].parameters;
            int[] activeOutcomes = params[pi].outcomes;
            for (int aoi = 0; aoi < activeOutcomes.length; aoi++) {
                if (useGaussianSmoothing) {
                    params[pi].parameters[aoi] += gaussianUpdate(pi, aoi, numEvents, correctionConstant);
                }
                else {
                    if (model[aoi] == 0) {
                        System.err.println("Model expects == 0 for " + predLabels[pi] + " " + outcomeLabels[aoi]);
                    }
                    // params[pi].updateParameter(aoi,(Math.log(observed[aoi]) -
                    // Math.log(model[aoi])));
                    params[pi].parameters[aoi] += (Math.log(observed[aoi]) - Math.log(model[aoi])) / correctionConstant;
                }
                modelExpects[pi].parameters[aoi] = 0.0; // re-initialize to 0.0's
            }
        }
        if (CFMOD > 0.0 && useSlackParameter)
            evalParams.setCorrectionParam(evalParams.getCorrectionParam() + (cfObservedExpect - Math.log(CFMOD)));

        display(". loglikelihood=" + loglikelihood + "\t" + ((double) numCorrect / numEvents) + "\n");
        return (loglikelihood);
    }

    private void display(String s) {
        if (printMessages) System.out.print(s);
    }

    /**
     * Creates a new <code>GISTrainer</code> instance.
     * 
     * @param printMessages sends progress messages about training to STDOUT when true; *
     * 
     * @param smoothingOb the "number" of times we want the trainer to imagine it saw a feature that
     *            it actually didn't see
     */
    public GISTrainer(boolean printMessages, boolean useSmoothing, double smoothingOb) {
        super();
        this.useSimpleSmoothing = useSmoothing;
        this._smoothingObservation = smoothingOb;
        this.printMessages = printMessages;
    }

    /**
     * Train a model using the GIS algorithm.
     * 
     * @param iterations The number of GIS iterations to perform.
     * @param indexer The object which will be used for event compilation.
     * @param printMessagesWhileTraining Determines whether training status messages are written to
     *            STDOUT.
     * @param smoothing Defines whether the created trainer will use smoothing while training the
     *            model.
     * @param SMOOTHING_OBSERVATION If we are using smoothing, this is used as the "number" of times
     *            we want the trainer to imagine that it saw a feature that it actually didn't see.
     *            Defaulted to 0.1.
     * @param modelPrior The prior distribution for the model.
     * @param cutoff The number of times a predicate must occur to be used in a model.
     * @return The newly trained model, which can be used immediately or saved to disk using an
     *         opennlp.maxent.io.GISModelWriter object.
     */
}