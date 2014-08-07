package main.java.org.maochen;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;

/**
 * Unigram word freq count
 * 
 * @author maochen
 * 
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
            this.hashCode();
        }

        public WordDatum addCount() {
            ++this.count;
            return this;
        }

        @Override
        public int compareTo(WordDatum o) {
            if (this.count > o.count)
                return -1;
            else if (this.count < o.count)
                return 1;
            else return 0;
        }
    }

    private Map<String, WordDatum> wordMap = new HashMap<String, WordDatum>();
    private static Queue<WordDatum> q = new PriorityQueue<WordDatum>();
    private int totalCount = 0;
    private boolean isNormalized = false;

    public synchronized void remove(String word) {
        int prevTotalCount = totalCount;
        if (wordMap.containsKey(word)) {
            WordDatum deletedWordDatum = wordMap.remove(word);
            totalCount -= isNormalized ? deletedWordDatum.getCount() * totalCount : deletedWordDatum.getCount();
            q.remove(deletedWordDatum);
        }

        if (isNormalized && prevTotalCount != totalCount) {
            reNormalize(prevTotalCount);
        }
    }

    public synchronized void put(String word) {
        if (word == null || word.isEmpty()) return;

        WordDatum wordObject = wordMap.containsKey(word) ? wordMap.get(word).addCount() : new WordDatum(word);
        if (!wordMap.containsKey(word)) wordMap.put(word, wordObject);

        // Do this Shit, or it wont do heapify automatically.
        if (q.contains(wordObject)) q.remove(wordObject);
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
        if (word == null || word.trim().isEmpty()) return null;
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

    public static void main(String[] args) {
        WordCount wordCount = new WordCount();
        wordCount.put("c");
        wordCount.put("a");
        wordCount.put("b");
        wordCount.put("b");
        wordCount.put("b");
        wordCount.put("a");
        for (int i = 0; i < 199; i++) {
            wordCount.put("ZZ");
        }
        wordCount.normalize();
        wordCount.normalize();
        wordCount.normalize();

        System.out.println(wordCount.getAllWords());

        wordCount.remove("ZZ");
        System.out.println(wordCount.getAllWords());
    }
}
