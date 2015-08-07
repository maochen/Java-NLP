package org.maochen.nlp.ml.datastructure;

import java.util.List;

/**
 * Created by Maochen on 8/5/15.
 */
public class SequenceTuple {
    public int id;
    public List<String> words;
    public List<String> tag;

    public SequenceTuple(List<String> words, List<String> tags) {
        if (words == null || tags == null || words.size() != tags.size()) {
            throw new RuntimeException("words and tags are invalid (Size mismatch).");
        }
        this.words = words;
        this.tag = tags;
    }

    public SequenceTuple() {

    }
}
