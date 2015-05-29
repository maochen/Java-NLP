package org.maochen.classifier.naivebayes;

import org.maochen.datastructure.LabelIndexer;

import java.io.Serializable;

/**
 * Created by Maochen on 5/29/15.
 */
public class NaiveBayesModel implements Serializable {
    // row=labelSize,col=featureLength
    double[][] meanVectors;
    double[][] varianceVectors;

    LabelIndexer labelIndexer;
}
