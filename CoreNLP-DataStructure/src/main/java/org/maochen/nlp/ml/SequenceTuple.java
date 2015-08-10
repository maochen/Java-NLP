package org.maochen.nlp.ml;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This tuple is for the Sequence Model using. Don't mess up with regular Tuple.
 *
 * Created by Maochen on 8/5/15.
 */
public class SequenceTuple {
    public int id;
    public List<String> words;
    public List<String> tag;

    private Map<String, Object> extra = null;

    public Map<String, Object> getExtra() {
        return extra;
    }

    public void addExtra(String key, Object val) {
        if (extra == null) {
            extra = new HashMap<>();
        }
        extra.put(key, val);
    }

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
