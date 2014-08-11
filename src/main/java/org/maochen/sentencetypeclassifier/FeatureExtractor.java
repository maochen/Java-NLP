package org.maochen.sentencetypeclassifier;

import com.clearnlp.constituent.CTLibEn;
import com.clearnlp.dependency.DEPLibEn;
import com.clearnlp.dependency.DEPNode;
import com.clearnlp.dependency.DEPTree;
import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Created by Maochen on 8/5/14.
 */
public class FeatureExtractor {

    private static final Logger LOG = LoggerFactory.getLogger(FeatureExtractor.class);

    private final String filepathPrefix;
    private String delimiter;
    private ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors(),
            new ThreadFactoryBuilder().setNameFormat("MaxEnt-FeatureExtractor-%d").build());

    private ClearNLPUtil parser;

    private boolean isRealFeature = false;

    // chunk, count
    private Map<String, Integer> biGram = new HashMap<>();

    private Map<String, Integer> triGram = new HashMap<>();

    public boolean getIsRealFeature() {
        return isRealFeature;
    }

    private void addFeats(StringBuilder builder, String key, Object value) {
        //        builder.append(key).append("=").append(value).append(delimiter);
        if ((Boolean) value) {
            builder.append(key).append(delimiter);
        }
    }

    // Bossssssss.... currently all binary features.
    private String generateFeats(String input) {
        StringBuilder builder = new StringBuilder();
        input = input.trim();
        input = input.replaceAll("_", " ");

        DEPTree tree = parser.process(input);

        // puncts.
        char punct = input.charAt(input.length() - 1);
        switch (punct) {
            case ';':
            case '.':
                addFeats(builder, "punct_dot", true);
                addFeats(builder, "punct_question", false);
                addFeats(builder, "punct_exclaim", false);
                break;
            case '!':
                addFeats(builder, "punct_dot", false);
                addFeats(builder, "punct_question", false);
                addFeats(builder, "punct_exclaim", true);
                break;
            case '?':
                addFeats(builder, "punct_dot", false);
                addFeats(builder, "punct_question", true);
                addFeats(builder, "punct_exclaim", false);
                break;
            default:
                addFeats(builder, "punct_dot", false);
                addFeats(builder, "punct_question", false);
                addFeats(builder, "punct_exclaim", false);
                break;
        }


        // keyword whether
        addFeats(builder, "whether", input.toLowerCase().contains("whether"));

        // 1st word POS
        Set<String> whPrefixPos = Sets.newHashSet(CTLibEn.POS_WRB, CTLibEn.POS_WDT, CTLibEn.POS_WP, CTLibEn.POS_WPS);
        String pos = tree.get(1).pos;
        addFeats(builder, "first_word_pos", whPrefixPos.contains(pos));

        // is 1st word rootVerb.
        addFeats(builder, "first_word_root_verb", pos.startsWith(CTLibEn.POS_VB));


        // Have aux in the sentence.
        int auxCount = Collections2.filter(tree, new Predicate<DEPNode>() {
            @Override
            public boolean apply(DEPNode depNode) {
                return DEPLibEn.DEP_AUX.equals(depNode.getLabel());
            }
        }).size();
        addFeats(builder, "aux_count", auxCount > 0);

        // Start with question word.
        Set<String> bagOfQuestionPrefix = Sets.newHashSet("tell me", "let me know", "clarify for me");
        boolean isStartPrefixMatch = false;
        for (String prefix : bagOfQuestionPrefix) {
            if (input.toLowerCase().startsWith(prefix)) {
                isStartPrefixMatch = true;
                break;
            }
        }
        addFeats(builder, "question_over_head", isStartPrefixMatch);

        // Verify - imperative
        addFeats(builder, "has_verify_keyword", "verify".equals(tree.get(1).form.toLowerCase()));

        String inputWithTag = " <sentence> " + input.toLowerCase() + " </sentence> ";
        inputWithTag = inputWithTag.replaceAll(" ", "_");
        //Bigram
        for (String str : biGram.keySet()) {
            // Make sure is the whole word match instead of partial word+"_"+partial word.
            addFeats(builder, "biGram_" + str, inputWithTag.contains("_" + str + "_"));
        }

        //Trigram
        for (String str : triGram.keySet()) {
            // Make sure is the whole word match instead of partial word+"_"+partial word.
            addFeats(builder, "triGram_" + str, inputWithTag.contains("_" + str + "_"));
        }
        return builder.toString().trim();
    }

    public String getFeats(String sentence) {
        String[] tokens = sentence.split(delimiter);
        if (tokens.length != 2) return "";

        StringBuilder builder = new StringBuilder();
        // Sentence
        builder.append(tokens[0]).append(delimiter);
        builder.append(generateFeats(tokens[0])).append(delimiter);
        // Label
        builder.append(tokens[1]);

        return builder.toString().trim();
    }

    public Set<String> getFeats(Set<String> trainEntries) {
        LOG.info("Extracting Features ...");

        final Set<String> vectorSet = Sets.newSetFromMap(new ConcurrentHashMap<String, Boolean>());

        LOG.info("Generating NGram Model ...");

        //Generate Bigram, Trigram
        for (String str : trainEntries) {
            // Grab the sentence.
            str = str.split(delimiter)[0];
            // Start End tag.
            str = "<sentence>_" + str.toLowerCase() + "_</sentence>";
            String[] tokens = str.split("_");
            for (int i = 0; i < tokens.length; i++) {
                if (i + 1 < tokens.length) {
                    String chunk = (tokens[i] + "_" + tokens[i + 1]);
                    int count = biGram.containsKey(chunk) ? biGram.get(chunk) : 0;
                    biGram.put(chunk, ++count);
                }
                if (i + 2 < tokens.length) {
                    String chunk = tokens[i] + "_" + tokens[i + 1] + "_" + tokens[i + 2];
                    int count = triGram.containsKey(chunk) ? triGram.get(chunk) : 0;
                    triGram.put(chunk, ++count);
                }
            }
        }

        // Delete these uncommon chunk
        biGram.values().removeAll(Sets.newHashSet(1));
        triGram.values().removeAll(Sets.newHashSet(1));
        // -----------

        // Persist for prediction use.
        persistNGram();
        LOG.info("NGram Model completed ...");

        List<Future<String>> futureList = new ArrayList<>();

        for (final String entry : trainEntries) {
            Callable<String> entryCallable = new Callable<String>() {
                @Override
                public String call() throws Exception {
                    String featVector = getFeats(entry);
                    vectorSet.add(featVector);
                    return featVector;
                }
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

    private void persistNGram() {
        String biGramFile = filepathPrefix + "/bigram";
        String triGramFile = filepathPrefix + "/trigram";

        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(biGramFile));
            oos.writeObject(biGram);
            oos.close();

            ObjectOutputStream oosTrigram = new ObjectOutputStream(new FileOutputStream(triGramFile));
            oosTrigram.writeObject(triGram);
            oosTrigram.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public FeatureExtractor(String filepathPrefix, String delimiter) {
        parser = new ClearNLPUtil();
        this.delimiter = delimiter;
        this.filepathPrefix = filepathPrefix;

        try {
            File bigramFile = new File(filepathPrefix + "/bigram");
            if (bigramFile.exists() && !bigramFile.isDirectory()) {
                ObjectInputStream ois = new ObjectInputStream(new FileInputStream(bigramFile));
                biGram = (Map) ois.readObject();
            }

            File trigramFile = new File(filepathPrefix + "/trigram");
            if (trigramFile.exists() && !trigramFile.isDirectory()) {
                ObjectInputStream ois = new ObjectInputStream(new FileInputStream(trigramFile));
                triGram = (Map) ois.readObject();
            }
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

}
