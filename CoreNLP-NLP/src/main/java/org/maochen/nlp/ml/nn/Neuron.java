package org.maochen.nlp.ml.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Created by Maochen on 3/2/16.
 */
public class Neuron {
    int id;
    int layer; // J
    List<Neuron> inputNeurons = new ArrayList<>();
    double[] inputVal;
    double[] theta; // from J to J+1
    Function<Double, Double> activationFunction;

    double[] outputVal;
    boolean isVisited = false;
    List<Neuron> outputNeurons = new ArrayList<>();

    public void process() {
        outputVal = new double[inputVal.length];
        for (int i = 0; i < inputVal.length; i++) {
            outputVal[i] = activationFunction.apply(inputVal[i]);
        }
        isVisited = true;
    }
}
