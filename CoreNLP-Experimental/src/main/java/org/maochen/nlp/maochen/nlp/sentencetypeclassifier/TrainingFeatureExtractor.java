package org.maochen.nlp.maochen.nlp.sentencetypeclassifier;

import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.maochen.nlp.datastructure.DTree;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Created by Maochen on 8/12/14.
 */
public class TrainingFeatureExtractor extends FeatureExtractor {
    private static final Logger LOG = LoggerFactory.getLogger(TrainingFeatureExtractor.class);

    private Map<String, DTree> depTreeCache = new HashMap<>();
    private ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors(),
            new ThreadFactoryBuilder().setNameFormat("MaxEnt-FeatureExtractor-%d").build());

    public TrainingFeatureExtractor(String filepathPrefix, String delimiter) {
        super(filepathPrefix, delimiter);
    }

    public void extractFeature(Set<String> trainingData) {
        File featureVectorFile = new File(filepathPrefix + "/featureVector.txt");
        if (!featureVectorFile.exists()) {
            try {
                featureVectorFile.createNewFile();

                FileWriter fw = new FileWriter(featureVectorFile.getAbsoluteFile());
                BufferedWriter bw = new BufferedWriter(fw);


                // annotatedTrainingData Delimiter
                Set<String> featureVector = getFeats(trainingData);
                for (String s : featureVector) {
                    bw.write(s + System.getProperty("line.separator"));
                }

                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void addToMap(Map<String, Integer> ngramMap, String... tokens) {
        String chunk = StringUtils.EMPTY;
        for (String token : tokens) {
            chunk = chunk + "_" + token;
        }
        chunk = chunk.substring(1);
        int count = ngramMap.containsKey(chunk) ? ngramMap.get(chunk) : 0;
        ngramMap.put(chunk, ++count);
    }

    // BFS
    private void generateDEPNGram(DTree tree) {
        String depString = super.getDEPString(tree);
        String[] depStringTokens = depString.split("_");

        for (int i = 0; i < depStringTokens.length; i++) {
            if (i + 1 < depStringTokens.length) {
                addToMap(biGramDepMap, depStringTokens[i], depStringTokens[i + 1]);
            }
            if (i + 2 < depStringTokens.length) {
                addToMap(triGramDepMap, depStringTokens[i], depStringTokens[i + 1], depStringTokens[i + 2]);
            }
        }
    }

    private void generateWordNGram(String sentence) {
        // Start End tag.
        String strWithStartEndTag = "<sentence>_" + sentence.toLowerCase() + "_</sentence>";
        String[] tokens = strWithStartEndTag.split("_");
        for (int i = 0; i < tokens.length; i++) {
            if (i + 1 < tokens.length) {
                addToMap(biGramWordMap, tokens[i], tokens[i + 1]);
            }
            if (i + 2 < tokens.length) {
                addToMap(triGramWordMap, tokens[i], tokens[i + 1], tokens[i + 2]);
            }
        }
    }

    private Set<String> getFeats(Set<String> trainEntries) {
        LOG.info("Extracting Features ...");

        final Set<String> vectorSet = Sets.newSetFromMap(new ConcurrentHashMap<String, Boolean>());

        LOG.info("Generating NGram Model ...");

        // Generate Bigram, Trigram
        for (String str : trainEntries) {
            // Grab the sentence.
            str = str.split(super.delimiter)[0];
            DTree tree = parser.parse(str.replaceAll("_", StringUtils.SPACE));
            depTreeCache.put(str, tree);
            generateDEPNGram(tree);
            generateWordNGram(str);
        }

        // Delete these uncommon chunk
        //        biGramWordMap.values().removeAll(Sets.newHashSet(1));
        //        triGramWordMap.values().removeAll(Sets.newHashSet(1));
        // -----------

        // Persist for prediction use.
        persistNGram();
        LOG.info("NGram Model completed ...");

        List<Future<String>> futureList = new ArrayList<>();

        final String delimiter = super.delimiter;
        for (final String entry : trainEntries) {
            Callable<String> entryCallable = () -> {
                String featVector = getFeats(entry, depTreeCache.get(entry.split(delimiter)[0]));
                vectorSet.add(featVector);
                return featVector;
            };

            Future<String> future = executorService.submit(entryCallable);
            futureList.add(future);
        }

        for (Future<String> future : futureList) {
            try {
                future.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        LOG.info("Extracting features completed.");
        return vectorSet;
    }

    private void serialize(String filepath, Map<String, Integer> dataMap) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filepath));
            oos.writeObject(dataMap);
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void persistNGram() {
        String biGramWordFile = super.filepathPrefix + "/bigram_word";
        String triGramWordFile = super.filepathPrefix + "/trigram_word";

        String biGramDepFile = super.filepathPrefix + "/bigram_dep";
        String triGramDepFile = super.filepathPrefix + "/trigram_dep";

        serialize(biGramWordFile, biGramWordMap);
        serialize(triGramWordFile, triGramWordMap);
        serialize(biGramDepFile, biGramDepMap);
        serialize(triGramDepFile, triGramDepMap);
    }
}
