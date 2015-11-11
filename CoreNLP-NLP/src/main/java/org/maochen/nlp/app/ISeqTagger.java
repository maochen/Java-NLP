package org.maochen.nlp.app;

import org.maochen.nlp.ml.SequenceTuple;

/**
 * Created by Maochen on 11/10/15.
 */
public interface ISeqTagger {
    void train(String trainFilePath);

    SequenceTuple predict(String sentence);

    void predict(SequenceTuple tuple);
}
