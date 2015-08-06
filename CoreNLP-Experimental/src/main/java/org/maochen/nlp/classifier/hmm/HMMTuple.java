package org.maochen.nlp.classifier.hmm;

import java.util.List;

/**
 * Created by Maochen on 8/5/15.
 */
public class HMMTuple {
    public List<String> words;
    public List<String> tag;

    public HMMTuple(List<String> words, List<String> tags) {
        if (words == null || tags == null || words.size() != tags.size()) {
            throw new RuntimeException("words and tags are invalid (Size mismatch).");
        }
        this.words = words;
        this.tag = tags;
    }

    public HMMTuple() {

    }
}
