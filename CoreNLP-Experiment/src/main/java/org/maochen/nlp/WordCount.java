package org.maochen.nlp;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;

/**
 * Unigram words freq count
 *
 * @author maochen
 */
public class WordCount {
    static class WordDatum implements Comparable<WordDatum> {
        final String word;
        // Might be probability later;
        double count;

        public String getWord() {
            return word;
        }

        public void setCount(double count) {
            this.count = count;
        }

        public double getCount() {
            return count;
        }

        public WordDatum(final String word) {
            this.word = word;
            count = 1;
        }

        public WordDatum addCount() {
            ++this.count;
            return this;
        }

        @Override
        public int compareTo(WordDatum o) {
            // Reverse
            return Double.compare(o.count, this.count);
        }
    }

    private Map<String, WordDatum> wordMap = new HashMap<>();
    private static final Queue<WordDatum> q = new PriorityQueue<>();
    private int totalCount = 0;
    private boolean isNormalized = false;

    public synchronized void remove(String word) {
        int prevTotalCount = totalCount;
        if (wordMap.containsKey(word)) {
            WordDatum deletedWordDatum = wordMap.remove(word);
            totalCount -= isNormalized ? deletedWordDatum.getCount() * totalCount :
                    deletedWordDatum.getCount();
            q.remove(deletedWordDatum);
        }

        if (isNormalized && prevTotalCount != totalCount) {
            reNormalize(prevTotalCount);
        }
    }

    public synchronized void put(String word) {
        if (word == null || word.isEmpty()) return;

        WordDatum wordObject = wordMap.containsKey(word) ?
                wordMap.get(word).addCount() : new WordDatum(word);
        if (!wordMap.containsKey(word)) {
            wordMap.put(word, wordObject);
        }

        // Do this Shit, or it wont do heapify automatically.
        if (q.contains(wordObject)) {
            q.remove(wordObject);
        }
        q.add(wordObject);
        ++totalCount;
    }

    public Map<String, Double> getTopX(final int x) {
        if (x <= 0) throw new RuntimeException("Top x is negative: " + x);

        Map<String, Double> result = new HashMap<>();
        int i = 0;

        synchronized (q) {
            Iterator<WordDatum> iter = q.iterator();
            while (iter.hasNext() && i++ < x) {
                WordDatum iterDatum = iter.next();
                result.put(iterDatum.getWord(), iterDatum.getCount());
            }
        }

        return result;
    }

    public WordDatum getWordDatum(final String word) {
        return wordMap.containsKey(word) ? wordMap.get(word) : null;
    }

    public Map<String, Double> getAllWords() {
        return getTopX(wordMap.size());
    }

    private void reNormalize(double previousTotalCount) {
        for (String key : wordMap.keySet()) {
            double normalizedFreq = wordMap.get(key).getCount() * previousTotalCount / totalCount;
            wordMap.get(key).setCount(normalizedFreq);
        }
    }

    public synchronized void normalize() {
        if (isNormalized) return;
        isNormalized = true;
        reNormalize(1.0);
    }
}
