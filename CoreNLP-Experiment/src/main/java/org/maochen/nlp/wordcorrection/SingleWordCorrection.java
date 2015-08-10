package org.maochen.nlp.wordcorrection;

import edu.stanford.nlp.ling.CoreLabel;
import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.parser.DoubleKeyMap;
import org.maochen.nlp.parser.stanford.StanfordParser;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * This is for the single words correction. It doesn't count in the prev/following words to generate NGram.
 */
public class SingleWordCorrection {
    static class Model implements Serializable {
        private static final long serialVersionUID = 1L;

        // Correct Word Dictionary
        public Map<String, Double> wordProbability = new HashMap<>();

        // public transient Table<String, String, Double> derivedWordProbability = HashBasedTable.create();

        // Correct,Derived,Probability
        // Row, Column, Value
        // Dont use Guava Table. Predict takes ~18ms vs. DoubleKeyMap Predict 3ms.
        public transient DoubleKeyMap<String, String, Double> derivedWordProbability = new DoubleKeyMap<>();

        public void normalizeWordProbability() {
            double totalCount = 0.0;
            for (String key : wordProbability.keySet()) {
                totalCount += wordProbability.get(key);
            }

            for (String key : wordProbability.keySet()) {
                double updatedCount = wordProbability.get(key) / totalCount;
                wordProbability.put(key, updatedCount);
            }

        }

        public void normalizeDerivedWordProbability() {
            for (String key1 : derivedWordProbability.rowKeySet()) {
                double totalCount = 0.0;
                Map<String, Double> k2map = derivedWordProbability.row(key1);
                for (String str : k2map.keySet()) {
                    totalCount += k2map.get(str);
                }

                for (String key2 : derivedWordProbability.row(key1).keySet()) {
                    derivedWordProbability.put(key1, key2, derivedWordProbability.get(key1, key2) / totalCount);
                }
            }
        }

        public void persist(String filename) throws IOException {
            FileOutputStream fos = new FileOutputStream(filename);
            ObjectOutputStream oos = new ObjectOutputStream(fos);

            oos.writeObject(this);
            oos.flush();
            oos.close();
        }

        public void restore(String filename) throws IOException, ClassNotFoundException {
            FileInputStream fis = new FileInputStream(filename);
            ObjectInputStream ois = new ObjectInputStream(fis);

            Model reload = (Model) ois.readObject();
            this.wordProbability = reload.wordProbability;
            ois.close();
        }
    }

    private Model model = new Model();

    // An edit can be a deletion (remove one letter), a transposition (swap adjacent letters), an
    // alteration (change one letter to another) or an insertion (add a letter).
    private Map<String, Double> distance1Generation(String word) {
        if (word == null || word.length() < 1) throw new RuntimeException("Input words Error: " + word);

        Map<String, Double> result = new HashMap<String, Double>();

        String prev;
        String last;

        for (int i = 0; i < word.length(); i++) {
            // Deletion
            prev = word.substring(0, i);
            last = word.substring(i + 1, word.length());
            result.put(prev + last, 1.0);

            // transposition
            if ((i + 1) < word.length()) {
                prev = word.substring(0, i);
                last = word.substring(i + 2, word.length());
                String trans = prev + word.charAt(i + 1) + word.charAt(i) + last;
                result.put(trans, 1.0);
            }

            // alter
            prev = word.substring(0, i);
            last = word.substring(i + 1, word.length());
            for (int j = 0; j < 26; j++) {
                result.put(prev + (char) (j + 97) + last, 1.0);
            }

            // insertion
            prev = word.substring(0, i);
            last = word.substring(i + 1, word.length());
            for (int j = 0; j < 26; j++) {
                result.put(prev + (char) (j + 97) + word.charAt(i) + last, 1.0);
                result.put(prev + word.charAt(i) + (char) (j + 97) + last, 1.0);
            }

        }

        result.remove(word);
        return result;
    }

    private Map<String, Double> errWordgenerating(String word) {
        Map<String, Double> oneDistance = distance1Generation(word);

        // Two Steps - One step seems makes more sense.
        //        Set<String> twoDistance = new HashSet<String>();
        //        for (String str : oneDistance.keySet()) {
        //            twoDistance.addAll(distance1Generation(str).keySet());
        //        }
        //
        //        for (String twoDistanceWord : twoDistance) {
        //            // 0.5 discounting needs to be discussed.
        //            if (!oneDistance.containsKey(twoDistanceWord)) {
        //                oneDistance.put(twoDistanceWord, 0.4);
        //            }
        //        }

        return oneDistance;
    }

    // Parser here is just for tokenize.
    public void buildModel(String wordFileName) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(new File(wordFileName)));

        String str;
        while ((str = br.readLine()) != null) {
            List<String> tokens = StanfordParser.stanfordTokenize(str).stream().map(CoreLabel::originalText).collect(Collectors.toList());
            for (String word : tokens) {
                double count = model.wordProbability.containsKey(word) ? model.wordProbability.get(word) : 0;
                count++;
                model.wordProbability.put(word, count);
            }
        }
        br.close();

        // Remove Empty prob.
        model.wordProbability.remove(StringUtils.EMPTY);
        model.normalizeWordProbability();
    }

    public String predict(String wrongWord) {
        if (model.wordProbability.containsKey(wrongWord)) return wrongWord;

        Map<String, Double> possibleCorrectWordMap = errWordgenerating(wrongWord);

        // Only take these in the dict into consideration
        // From possible words generate all err words.
        // Remove these non-err words.
        // Dont use for-loop, far more slower.
        possibleCorrectWordMap.keySet().stream()
                .filter(model.wordProbability::containsKey)
                .forEach(possibleCorrectWord -> {
                    // From possible words generate all err words.
                    Map<String, Double> errWordMap = errWordgenerating(possibleCorrectWord);

                    // Remove these non-err words.
                    // Dont use for-loop, far more slower.
                    errWordMap.keySet().removeAll(model.wordProbability.keySet());

                    for (String errWord : errWordMap.keySet()) {
                        model.derivedWordProbability.put(possibleCorrectWord, errWord, errWordMap.get(errWord));
                    }

                });

        if (model.derivedWordProbability.size() == 0) {
            throw new RuntimeException("No Correction Suggestion");
        }

        model.normalizeDerivedWordProbability();

        double argmaxProb = 0.0;
        String argmaxWord = StringUtils.EMPTY;
        // P(w|c) Map - <correct,value>
        Map<String, Double> pwc = model.derivedWordProbability.column(wrongWord);
        for (String correctWord : pwc.keySet()) {
            System.out.println("[predict] Possible: " + correctWord + "=" + pwc.get(correctWord) + "\t|\twordProb: " + model.wordProbability.get(correctWord));
            double localarg = pwc.get(correctWord) * model.wordProbability.get(correctWord);
            if (localarg >= argmaxProb) {
                argmaxProb = localarg;
                argmaxWord = correctWord;
            }
        }

        return argmaxWord;
    }

    public void persistModel(String filename) throws IOException {
        model.persist(filename);
    }

    public void restoreModel(String filename) throws IOException, ClassNotFoundException {
        model.restore(filename);
    }

    public SingleWordCorrection() {
        if (model == null) {
            model = new Model();
        }
    }

    public static void main(String[] args) throws IOException {
        String path = SingleWordCorrection.class.getClassLoader().getResource("the_adventures_of_sherlock_holmes.txt").getFile();
        SingleWordCorrection swc = new SingleWordCorrection();
        swc.buildModel(path);
        // swc.persistModel("model.dat");
        // swc.restoreModel("model.dat");
        String word = "prob";
        Long start = System.currentTimeMillis();
        String predict = swc.predict(word);
        Long end = System.currentTimeMillis();

        System.out.println(word + "->" + predict);

        System.out.println("\nPredict Time Elapse: " + (end - start) + "ms");
    }
}
