package org.maochen.nlp.ml.classifier.maxent;

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import opennlp.maxent.GISModel;
import opennlp.model.DataIndexer;
import opennlp.model.EvalParameters;
import opennlp.model.MutableContext;
import opennlp.model.Prior;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


/**
 * An implementation of Generalized Iterative Scaling.  The reference paper for this implementation
 * was Adwait Ratnaparkhi's tech report at the University of Pennsylvania's Institute for Research
 * in Cognitive Science, and is available at <a href ="ftp://ftp.cis.upenn.edu/pub/ircs/tr/97-08.ps.Z"><code>ftp://ftp.cis.upenn.edu/pub/ircs/tr/97-08.ps.Z</code></a>.
 *
 * The slack parameter used in the above implementation has been removed by default from the
 * computation and a method for updating with Gaussian smoothing has been added per Investigating
 * GIS and Smoothing for Maximum Entropy Taggers, Clark and Curran (2002). <a
 * href="http://acl.ldc.upenn.edu/E/E03/E03-1071.pdf"><code>http://acl.ldc.upenn.edu/E/E03/E03-1071.pdf</code></a>
 * The slack parameter can be used by setting <code>useSlackParameter</code> to true. Gaussian
 * smoothing can be used by setting <code>useGaussianSmoothing</code> to true.
 *
 * A prior can be used to train models which converge to the distribution which minimizes the
 * relative entropy between the distribution specified by the empirical constraints of the training
 * data and the specified prior.  By default, the uniform distribution is used as the prior.
 */
class GISTrainer {

    private static final Logger LOG = LoggerFactory.getLogger(GISTrainer.class);

    /**
     * Specifies whether unseen context/outcome pairs should be estimated as occur very
     * infrequently.
     */
    private boolean useSimpleSmoothing = false;

    /**
     * Specified whether parameter updates should prefer a distribution of parameters which is
     * gaussian.
     */
    private boolean useGaussianSmoothing = false;

    private double sigma = 2.0;

    // If we are using smoothing, this is used as the "number" of
    // times we want the trainer to imagine that it saw a feature that it
    // actually didn't see.  Defaulted to 0.1.
    private double smoothingObservation = 0.1;

    /**
     * Number of unique events which occured in the event set.
     */
    private int numUniqueEvents;

    /**
     * Records the array of predicates seen in each event.
     */
    private int[][] trainingDataFeatNameIndices;

    /**
     * The value associated with each context. If null then context values are assumes to be 1.
     */
    private float[][] trainingDataFeatValues;

    /**
     * List of outcomes for each event i, in context[i].
     */
    private int[] outcomeList;

    /**
     * Records the num of times an event has been seen for each event i, in context[i].
     */
    private int[] numTimesEventsSeen;

    /**
     * The number of times a predicate occured in the training data.
     */
    private int[] predicateCounts;

    private int cutoff;

    /**
     * Stores the String names of the outcomes. The GIS only tracks outcomes as ints, and so this
     * array is needed to save the model to disk and thereby allow users to know what the outcome
     * was in human understandable terms.
     */
    private String[] labels;

    /**
     * Stores the String names of the predicates. The GIS only tracks predicates as ints, and so
     * this array is needed to save the model to disk and thereby allow users to know what the
     * outcome was in human understandable terms.
     */
    private String[] featNames;

    /**
     * Stores the observed expected values of the features based on training data.
     */
    private MutableContext[] observedExpects;

    /**
     * Stores the estimated parameter value of each predicate during iteration
     */
    private MutableContext[] params;

    /**
     * Stores the expected values of the features based on the current models
     */
    private MutableContext[][] modelExpects;

    /**
     * This is the prior distribution that the model uses for training.
     */
    private Prior prior;

    private static final double LLThreshold = 0.0001;

    /**
     * Initial probability for all outcomes.
     */
    private EvalParameters evalParams;

    /**
     * Sets whether this trainer will use smoothing while training the model. This can improve model
     * accuracy, though training will potentially take longer and use more memory.  Model size will
     * also be larger.
     *
     * @param smooth true if smoothing is desired, false if not
     */
    public void setSmoothing(boolean smooth) {
        useSimpleSmoothing = smooth;
    }

    /**
     * Sets whether this trainer will use smoothing while training the model. This can improve model
     * accuracy, though training will potentially take longer and use more memory.  Model size will
     * also be larger.
     *
     * @param timesSeen the "number" of times we want the trainer to imagine it saw a feature that
     *                  it actually didn't see
     */
    public void setSmoothingObservation(double timesSeen) {
        smoothingObservation = timesSeen;
    }

    /**
     * Sets whether this trainer will use smoothing while training the model. This can improve model
     * accuracy, though training will potentially take longer and use more memory.  Model size will
     * also be larger.
     */
    public void setGaussianSigma(double sigmaValue) {
        useGaussianSmoothing = true;
        sigma = sigmaValue;
    }


    /**
     * Train a model using the GIS algorithm.
     *
     * @param iterations The number of GIS iterations to perform.
     * @param di         The data indexer used to compress events in memory.
     * @param modelPrior The prior distribution used to train this model.
     * @return The newly trained model, which can be used immediately or saved to disk using an
     * opennlp.maxent.io.GISModelWriter object.
     */
    public GISModel trainModel(int iterations, DataIndexer di, Prior modelPrior, int cutoff, int threads) {
        if (threads <= 0) {
            threads = 1;
        }

        modelExpects = new MutableContext[threads][];

        /************** Incorporate all of the needed info ******************/
        LOG.debug("Incorporating indexed data for training...");
        trainingDataFeatNameIndices = di.getContexts();
        trainingDataFeatValues = di.getValues();
        this.cutoff = cutoff;
        predicateCounts = di.getPredCounts();
        numTimesEventsSeen = di.getNumTimesEventsSeen();
        numUniqueEvents = trainingDataFeatNameIndices.length;
        this.prior = modelPrior;

        labels = di.getOutcomeLabels();
        outcomeList = di.getOutcomeList();

        featNames = di.getPredLabels();
        prior.setLabels(labels, featNames);

        // determine the correction constant and its inverse
        double correctionConstant = 0;
        for (int ci = 0; ci < trainingDataFeatNameIndices.length; ci++) {
            if (trainingDataFeatValues == null || trainingDataFeatValues[ci] == null) {
                if (trainingDataFeatNameIndices[ci].length > correctionConstant) {
                    correctionConstant = trainingDataFeatNameIndices[ci].length;
                }
            } else {
                float cl = trainingDataFeatValues[ci][0];
                for (int vi = 1; vi < trainingDataFeatValues[ci].length; vi++) {
                    cl += trainingDataFeatValues[ci][vi];
                }

                if (cl > correctionConstant) {
                    correctionConstant = cl;
                }
            }
        }

        LOG.debug("Number of Event Tokens: " + numUniqueEvents);
        LOG.debug("Number of Outcomes: " + labels.length);
        LOG.debug("Number of Predicates: " + featNames.length);

        // set up feature arrays
        float[][] featCount = new float[featNames.length][labels.length];
        for (int ti = 0; ti < numUniqueEvents; ti++) {
            for (int j = 0; j < trainingDataFeatNameIndices[ti].length; j++) {
                if (trainingDataFeatValues != null && trainingDataFeatValues[ti] != null) {
                    featCount[trainingDataFeatNameIndices[ti][j]][outcomeList[ti]] += numTimesEventsSeen[ti] * trainingDataFeatValues[ti][j];
                } else {
                    featCount[trainingDataFeatNameIndices[ti][j]][outcomeList[ti]] += numTimesEventsSeen[ti];
                }
            }
        }

        // A fake "observation" to cover features which are not detected in
        // the data.  The default is to assume that we observed "1/10th" of a
        // feature during training.
        final double smoothingObservation = this.smoothingObservation;

        // Get the observed expectations of the features. Strictly speaking,
        // we should divide the counts by the number of Tokens, but because of
        // the way the model's expectations are approximated in the
        // implementation, this is cancelled out when we compute the next
        // iteration of a parameter, making the extra divisions wasteful.
        params = new MutableContext[featNames.length];
        for (int i = 0; i < modelExpects.length; i++) {
            modelExpects[i] = new MutableContext[featNames.length];
        }
        observedExpects = new MutableContext[featNames.length];

        // The model does need the correction constant and the correction feature. The correction constant
        // is only needed during training, and the correction feature is not necessary.
        // For compatibility reasons the model contains form now on a correction constant of 1,
        // and a correction param 0.
        evalParams = new EvalParameters(params, 0, 1, labels.length);
        int[] activeOutcomes = new int[labels.length];
        int[] labelPattern = new int[labels.length];
        int[] outcomePattern;

        for (int oi = 0; oi < labels.length; oi++) {
            labelPattern[oi] = oi;
        }

        for (int pi = 0; pi < featNames.length; pi++) {
            int numActiveOutcomes = 0;
            if (useSimpleSmoothing) {
                numActiveOutcomes = labels.length;
                outcomePattern = labelPattern;
            } else { //determine active outcomes
                for (int oi = 0; oi < labels.length; oi++) {
                    if (featCount[pi][oi] > 0 && predicateCounts[pi] >= cutoff) {
                        activeOutcomes[numActiveOutcomes] = oi;
                        numActiveOutcomes++;
                    }
                }
                if (numActiveOutcomes == labels.length) {
                    outcomePattern = labelPattern;
                } else {
                    outcomePattern = new int[numActiveOutcomes];
                    for (int aoi = 0; aoi < numActiveOutcomes; aoi++) {
                        outcomePattern[aoi] = activeOutcomes[aoi];
                    }
                }
            }
            params[pi] = new MutableContext(outcomePattern, new double[numActiveOutcomes]);
            for (int i = 0; i < modelExpects.length; i++)
                modelExpects[i][pi] = new MutableContext(outcomePattern, new double[numActiveOutcomes]);
            observedExpects[pi] = new MutableContext(outcomePattern, new double[numActiveOutcomes]);
            for (int aoi = 0; aoi < numActiveOutcomes; aoi++) {
                int oi = outcomePattern[aoi];
                params[pi].setParameter(aoi, 0.0);
                for (MutableContext[] modelExpect : modelExpects) {
                    modelExpect[pi].setParameter(aoi, 0.0);
                }
                if (featCount[pi][oi] > 0) {
                    observedExpects[pi].setParameter(aoi, featCount[pi][oi]);
                } else if (useSimpleSmoothing) {
                    observedExpects[pi].setParameter(aoi, smoothingObservation);
                }
            }
        }

        /***************** Find the parameters ************************/
        LOG.debug("Computing model parameters in " + threads + " threads...");
        findParameters(iterations, correctionConstant);

        /*************** Create and return the model ******************/
        // To be compatible with old models the correction constant is always 1
        return new GISModel(params, featNames, labels, 1, evalParams.getCorrectionParam());

    }

    /* Estimate and return the model parameters. */
    private void findParameters(int iterations, double correctionConstant) {
        LOG.info("Performing max " + iterations + " iterations.");

        double prevLL = 0.0;
        for (int i = 1; i <= iterations; i++) {
            LOG.info("Iteration " + i);
            double currLL = nextIteration(correctionConstant); // Core

            if (i > 1) {
                if (prevLL > currLL) {
                    LOG.error("Model Diverging: loglikelihood decreased");
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
        trainingDataFeatNameIndices = null;
    }

    //modeled on implementation in  Zhang Le's maxent kit
    private double gaussianUpdate(int predicate, int oid, int n, double correctionConstant) {
        double param = params[predicate].getParameters()[oid];
        double x0 = 0.0;
        double modelValue = modelExpects[0][predicate].getParameters()[oid];
        double observedValue = observedExpects[predicate].getParameters()[oid];
        for (int i = 0; i < 50; i++) {
            double tmp = modelValue * Math.exp(correctionConstant * x0);
            double f = tmp + (param + x0) / sigma - observedValue;
            double fp = tmp * correctionConstant + 1 / sigma;
            if (fp == 0) {
                break;
            }
            double x = x0 - f / fp;
            if (Math.abs(x - x0) < 0.000001) {
                x0 = x;
                break;
            }
            x0 = x;
        }
        return x0;
    }

    private class ModelExpactationComputeTask implements Callable<ModelExpactationComputeTask> {

        private final int startIndex;
        private final int length;

        private double loglikelihood = 0;

        private int numEvents = 0;
        private int numCorrect = 0;

        final private int threadIndex;

        // startIndex to compute, number of events to compute
        ModelExpactationComputeTask(int threadIndex, int startIndex, int length) {
            this.startIndex = startIndex;
            this.length = length;
            this.threadIndex = threadIndex;
        }

        public ModelExpactationComputeTask call() {

            final double[] modelDistribution = new double[labels.length];

            for (int ei = startIndex; ei < startIndex + length; ei++) {
                prior.logPrior(modelDistribution, trainingDataFeatNameIndices[ei], trainingDataFeatValues[ei]);
                GISModel.eval(trainingDataFeatNameIndices[ei], trainingDataFeatValues[ei], modelDistribution, evalParams);

                for (int j = 0; j < trainingDataFeatNameIndices[ei].length; j++) {
                    int pi = trainingDataFeatNameIndices[ei][j];
                    if (predicateCounts[pi] < cutoff) {
                        continue;
                    }

                    int[] activeOutcomes = modelExpects[threadIndex][pi].getOutcomes();
                    for (int aoi = 0; aoi < activeOutcomes.length; aoi++) {
                        int oi = activeOutcomes[aoi];

                        // numTimesEventsSeen must also be thread safe
                        if (trainingDataFeatValues != null && trainingDataFeatValues[ei] != null) {
                            modelExpects[threadIndex][pi].updateParameter(aoi, modelDistribution[oi] * trainingDataFeatValues[ei][j] * numTimesEventsSeen[ei]);
                        } else {
                            modelExpects[threadIndex][pi].updateParameter(aoi, modelDistribution[oi] * numTimesEventsSeen[ei]);
                        }
                    }

                }

                loglikelihood += Math.log(modelDistribution[outcomeList[ei]]) * numTimesEventsSeen[ei];

                numEvents += numTimesEventsSeen[ei];
                if (LOG.isDebugEnabled()) {
                    int maxIndex = 0;
                    for (int labelIndex = 1; labelIndex < labels.length; labelIndex++) {
                        if (modelDistribution[labelIndex] > modelDistribution[maxIndex]) {
                            maxIndex = labelIndex;
                        }
                    }
                    if (maxIndex == outcomeList[ei]) {
                        numCorrect += numTimesEventsSeen[ei];
                    }
                }

            }

            return this;
        }

        synchronized int getNumEvents() {
            return numEvents;
        }

        synchronized int getNumCorrect() {
            return numCorrect;
        }

        synchronized double getLoglikelihood() {
            return loglikelihood;
        }
    }

    /* Compute one iteration of GIS and return log-likelihood.*/
    private double nextIteration(double correctionConstant) {
        // compute contribution of p(a|b_i) for each feature and the new
        // correction parameter
        double loglikelihood = 0.0;
        int numEvents = 0;
        int numCorrect = 0;

        int numberOfThreads = modelExpects.length;
        ExecutorService executor = Executors.newFixedThreadPool(numberOfThreads);
        int taskSize = numUniqueEvents / numberOfThreads;
        int leftOver = numUniqueEvents % numberOfThreads;

        List<Future<?>> futures = new ArrayList<Future<?>>();

        for (int i = 0; i < numberOfThreads; i++) {
            if (i != numberOfThreads - 1)
                futures.add(executor.submit(new ModelExpactationComputeTask(i, i * taskSize, taskSize)));
            else
                futures.add(executor.submit(new ModelExpactationComputeTask(i, i * taskSize, taskSize + leftOver)));
        }

        for (Future<?> future : futures) {
            ModelExpactationComputeTask finishedTask;
            try {
                finishedTask = (ModelExpactationComputeTask) future.get();
            } catch (InterruptedException e) {
                // TODO: We got interrupted, but that is currently not really supported!
                // For now we just print the exception and fail hard. We hopefully soon
                // handle this case properly!
                e.printStackTrace();
                throw new IllegalStateException("Interruption is not supported!", e);
            } catch (ExecutionException e) {
                // Only runtime exception can be thrown during training, if one was thrown
                // it should be re-thrown. That could for example be a NullPointerException
                // which is caused through a bug in our implementation.
                throw new RuntimeException("Exception during training: " + e.getMessage(), e);
            }

            // When they are done, retrieve the results ...
            numEvents += finishedTask.getNumEvents();
            numCorrect += finishedTask.getNumCorrect();
            loglikelihood += finishedTask.getLoglikelihood();
        }

        executor.shutdown();

        // merge the results of the two computations
        for (int pi = 0; pi < featNames.length; pi++) {
            int[] activeOutcomes = params[pi].getOutcomes();

            for (int aoi = 0; aoi < activeOutcomes.length; aoi++) {
                for (int i = 1; i < modelExpects.length; i++) {
                    modelExpects[0][pi].updateParameter(aoi, modelExpects[i][pi].getParameters()[aoi]);
                }
            }
        }

        // compute the new parameter values
        for (int pi = 0; pi < featNames.length; pi++) {
            double[] observed = observedExpects[pi].getParameters();
            double[] model = modelExpects[0][pi].getParameters();
            int[] activeOutcomes = params[pi].getOutcomes();
            for (int aoi = 0; aoi < activeOutcomes.length; aoi++) {
                if (useGaussianSmoothing) {
                    params[pi].updateParameter(aoi, gaussianUpdate(pi, aoi, numEvents, correctionConstant));
                } else {
                    if (model[aoi] == 0) {
                        LOG.error("Model expects == 0 for " + featNames[pi] + " " + labels[aoi]);
                    }
                    params[pi].updateParameter(aoi, ((Math.log(observed[aoi]) - Math.log(model[aoi])) / correctionConstant));
                }

                for (MutableContext[] modelExpect : modelExpects) {
                    modelExpect[pi].setParameter(aoi, 0.0); // re-initialize to 0.0's
                }

            }
        }

        LOG.info("loglikelihood = " + loglikelihood + "\taccurancy:\t" + ((double) numCorrect / numEvents));

        return loglikelihood;
    }
}
