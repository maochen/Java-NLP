package org.maochen.sentencetypeclassifier;

import com.clearnlp.constituent.CTLibEn;
import com.clearnlp.dependency.DEPLibEn;
import com.clearnlp.dependency.DEPNode;
import com.clearnlp.dependency.DEPTree;
import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import com.google.common.collect.Sets;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.*;

/**
 * Created by Maochen on 8/5/14.
 */
public class FeatureExtractor {

    private static final Logger LOG = LoggerFactory.getLogger(FeatureExtractor.class);

    final String filepathPrefix;

    ClearNLPUtil parser;

    String delimiter;

    boolean isRealFeature = false;

    // chunk, count
    Map<String, Integer> biGramWordMap = new HashMap<>();
    Map<String, Integer> triGramWordMap = new HashMap<>();

    // label, count
    Map<String, Integer> biGramDepMap = new HashMap<>();
    Map<String, Integer> triGramDepMap = new HashMap<>();

    private void addFeats(StringBuilder builder, String key, Object value, int weight) {
        //        builder.append(key).append("=").append(value).append(delimiter);
        if ((Boolean) value) {
            for (int i = 0; i < weight; i++) {
                builder.append(key).append(delimiter);
            }
        }
    }

    protected String getDEPString(DEPTree tree) {
        StringBuilder builder = new StringBuilder();
        builder.append("_<DEP>_");
        Queue<DEPNode> q = new LinkedList<>();
        q.add(tree.getFirstRoot());

        while (!q.isEmpty()) {
            DEPNode currentNode = q.poll();
            if (currentNode == null) {
                continue;
            }
            builder.append(currentNode.getLabel()).append("_");
            for (DEPNode child : currentNode.getDependentNodeList()) {
                q.add(child);
            }
        }

        builder.append("</DEP>_");
        return builder.toString();
    }

    // Bossssssss.... currently all binary features.
    private String generateFeats(String sentence, DEPTree tree) {
        StringBuilder builder = new StringBuilder();
        sentence = sentence.trim();
        sentence = sentence.replaceAll("_", " ");
        int sentenceLength = sentence.split("\\s").length;
        int weight = sentenceLength > 10 ? sentenceLength : 10;

        String inputWithTag = sentence.toLowerCase();
        // Remove Punct at the end for NGram
        inputWithTag = inputWithTag.replaceAll("\\p{Punct}*$", "");
        inputWithTag = " <sentence> " + inputWithTag + " </sentence> ";
        inputWithTag = inputWithTag.replaceAll(" ", "_");
        // Bigram
        for (String str : biGramWordMap.keySet()) {
            // Make sure is the whole word match instead of partial word+"_"+partial word.
            addFeats(builder, "biGramWord_" + str, inputWithTag.contains("_" + str + "_"), 1);
        }

        // Trigram
        for (String str : triGramWordMap.keySet()) {
            // Make sure is the whole word match instead of partial word+"_"+partial word.
            addFeats(builder, "triGramWord_" + str, inputWithTag.contains("_" + str + "_"), 1);
        }

        String depString = getDEPString(tree);
        for (String str : biGramDepMap.keySet()) {
            addFeats(builder, "biGramDEP_" + str, depString.contains("_" + str + "_"), 1);
        }

        for (String str : triGramDepMap.keySet()) {
            addFeats(builder, "triGramDEP_" + str, depString.contains("_" + str + "_"), 1);
        }

        Set<String> whPrefixPos = Sets.newHashSet(CTLibEn.POS_WRB, CTLibEn.POS_WDT, CTLibEn.POS_WP, CTLibEn.POS_WPS);
        // 1st word is WH
        String firstPOS = tree.get(1).pos;
        addFeats(builder, "first_word_pos", whPrefixPos.contains(firstPOS), 1);

        // last word is WH
        int lastPOSIndex = sentence.matches(".*\\p{Punct}$") ? tree.size() - 2 : tree.size() - 1;
        String lastPOS = tree.get(lastPOSIndex).pos;
        addFeats(builder, "last_word_pos", whPrefixPos.contains(lastPOS), 1);

        // is 1st word rootVerb.
        addFeats(builder, "first_word_root_verb", firstPOS.startsWith(CTLibEn.POS_VB) && tree.get(1).isRoot(), weight);

        // Have aux in the sentence.
        int auxCount = Collections2.filter(tree, new Predicate<DEPNode>() {
            @Override
            public boolean apply(DEPNode depNode) {
                return DEPLibEn.DEP_AUX.equals(depNode.getLabel());
            }
        }).size();
        addFeats(builder, "has_aux", auxCount > 0, 1);

        // Start with question word.
        Set<String> bagOfQuestionPrefix = Sets.newHashSet("tell me", "let me know", "clarify for me", "name");
        boolean isStartPrefixMatch = false;
        for (String prefix : bagOfQuestionPrefix) {
            if (sentence.toLowerCase().startsWith(prefix)) {
                isStartPrefixMatch = true;
                break;
            }
        }
        addFeats(builder, "question_over_head", isStartPrefixMatch, 1);

        // Verify, Ask, Say - imperative
        Set<String> imperativeKeywords = Sets.newHashSet("verify", "ask", "say", "solve", "run", "execute");
        boolean isImperativeStart = imperativeKeywords.contains(tree.get(1).lemma) && tree.get(1).isRoot();
        addFeats(builder, "has_imperative_keyword", isImperativeStart, weight);

        // puncts.
        char punct = sentence.charAt(sentence.length() - 1);
        switch (punct) {
            case ';':
            case '.':
                addFeats(builder, "punct_dot", true, 1);
                addFeats(builder, "punct_question", false, 1);
                addFeats(builder, "punct_exclaim", false, 1);
                break;
            case '!':
                addFeats(builder, "punct_dot", false, 1);
                addFeats(builder, "punct_question", false, 1);
                addFeats(builder, "punct_exclaim", true, weight);
                break;
            case '?':
                // Just give more weights for ?
                addFeats(builder, "punct_dot", false, 1);
                addFeats(builder, "punct_question", true, weight);
                addFeats(builder, "punct_exclaim", false, 1);
                break;
            default:
                addFeats(builder, "punct_dot", false, 1);
                addFeats(builder, "punct_question", false, 1);
                addFeats(builder, "punct_exclaim", false, 1);
                break;
        }

        // keyword whether
        addFeats(builder, "whether", sentence.toLowerCase().contains("whether"), 1);

        return builder.toString().trim();
    }

    public String getFeats(String entry) {
        DEPTree tree = parser.process(entry.split(delimiter)[0].replaceAll("_", " "));
        return getFeats(entry, tree);
    }

    public String getFeats(String entry, DEPTree tree) {
        String[] tokens = entry.split(delimiter);
        if (tokens.length != 2) return "";

        StringBuilder builder = new StringBuilder();
        // Sentence
        builder.append(tokens[0]).append(delimiter);
        builder.append(generateFeats(tokens[0], tree)).append(delimiter);
        // Label
        builder.append(tokens[1]);

        return builder.toString().trim();
    }

    private Map<String, Integer> deserialize(String filePath) {
        try {
            File serializedFile = new File(filePath);
            if (serializedFile.exists() && !serializedFile.isDirectory()) {
                ObjectInputStream ois = new ObjectInputStream(new FileInputStream(serializedFile));
                return (Map) ois.readObject();
            }
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        return new HashMap<>();
    }

    public FeatureExtractor(String filepathPrefix, String delimiter) {
        this.delimiter = delimiter;

        parser = new ClearNLPUtil();
        this.filepathPrefix = filepathPrefix;

        biGramWordMap = deserialize(filepathPrefix + "/bigram_word");
        triGramWordMap = deserialize(filepathPrefix + "/trigram_word");
        biGramDepMap = deserialize(filepathPrefix + "/bigram_dep");
        triGramDepMap = deserialize(filepathPrefix + "/trigram_dep");
    }
}
