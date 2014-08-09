package org.maochen.sentencetypeclassifier;

import com.clearnlp.component.util.CSenTypeClassifierEN;
import com.clearnlp.constituent.CTLibEn;
import com.clearnlp.dependency.DEPLibEn;
import com.clearnlp.dependency.DEPNode;
import com.clearnlp.dependency.DEPTree;
import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;

import java.io.*;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by Maochen on 8/6/14.
 */
public class ClearNLPSentenceTagger {
    ClearNLPUtil parser;

    private boolean isWhatNPExpression(DEPTree tree) {
        // If user has marked the sentence as a question, it's a question.
        if (tree.toStringRaw().endsWith("?")) {
            return false;
        }

        // See if the parse matches the "What + NounPhrase" pattern.
        DEPNode root = tree.getFirstRoot();
        String posRoot = root.pos;
        // If the root is a noun phrase...
        if (posRoot.equals(CTLibEn.POS_NN) || posRoot.equals(CTLibEn.POS_NNS)) {
            // If noun phrase has "what" as a dependency with a det dependency relation and a WP/WDT pos...
            Collection<DEPNode> detNodes = Collections2.filter(tree, new Predicate<DEPNode>() {
                @Override
                public boolean apply(DEPNode depNode) {
                    return DEPLibEn.DEP_DET.equals(depNode.getLabel());
                }
            });
            if (detNodes != null) {
                for (DEPNode detNode : detNodes) {
                    String posDet = detNode.pos;
                    if (posDet.equals(CTLibEn.POS_WP) || posDet.equals(CTLibEn.POS_WDT)) {
                        if (detNode.lemma.toLowerCase().equals("what")) {
                            return true;
                        }
                    }
                }
            }
        }


        return false;
    }

    private boolean isHowAdjExpression(DEPTree tree) {
        // If user has marked the sentence as a question, it's a question.
        if (tree.toStringRaw().endsWith("?")) {
            return false;
        }
        // See if the parse matches the "How + Adjective" pattern.
        DEPNode root = tree.getFirstRoot();
        String posRoot = root.pos;
        // If the root is an ajective or past participle...
        // (We're ignoring POS_JJR and POS_JJS (comparative and superlative).
        // We need POS_RB because parser sometimes mislabels adjectives as adverbs, e.g. "How pretty.".)
        if (posRoot.equals(CTLibEn.POS_JJ) || posRoot.equals(CTLibEn.POS_VBN) || posRoot.equals(CTLibEn.POS_RB)) {
            // If adjective has "how" as a dependency with an advmod dependency relation and a WRB pos...
            Collection<DEPNode> detNodes = Collections2.filter(tree, new Predicate<DEPNode>() {
                @Override
                public boolean apply(DEPNode depNode) {
                    return DEPLibEn.DEP_AMOD.equals(depNode.getLabel());
                }
            });
            if (detNodes != null) {
                for (DEPNode detNode : detNodes) {
                    String posDet = detNode.pos;
                    if (posDet.equals(CTLibEn.POS_WRB)) {
                        if (detNode.lemma.toLowerCase().equals("how")) {
                            return true;
                        }
                    }
                }
            }
        }


        return false;
    }

    private boolean isHereBeExpression(DEPTree tree) {
        if (tree.get(1).lemma.equalsIgnoreCase("here")) return true;
        return false;
    }

    public String identifySentenceType(String sentence) {
        DEPTree tree = parser.process(sentence);
        CSenTypeClassifierEN stc = new CSenTypeClassifierEN();
        String type = "";

        for (int i = 1; i < tree.size(); i++) {
            DEPNode node = tree.get(i);

            if (i == 1 && "be".equalsIgnoreCase(node.lemma) && !"be".equalsIgnoreCase(node.form)) {
                type = "interrogative";
                break;
            } else if (node.isRoot()) {
                if (stc.isInterrogative(node)) {
                    if (isWhatNPExpression(tree) || isHowAdjExpression(tree) || isHereBeExpression(tree)) {
                        type = "declarative";
                    } else {
                        type = "interrogative";
                    }
                } else if (i == 1 || stc.isImperative(node)) {
                    type = "imperative";
                }
            }
        }

        type = type.isEmpty() ? "declarative" : type;
        return type;
    }

    public void processText(String trainFilePath, String outputPath) throws IOException {
        Set<String> history = new HashSet<>();

        StringBuilder builder = new StringBuilder();

        try (BufferedReader br = new BufferedReader(new FileReader(trainFilePath))) {
            String line = br.readLine();
            while (line != null) {
                if (!history.contains(line)) {
                    String type = identifySentenceType(line);
                    builder.append(line).append("\t").append(type).append("\n");
                    history.add(line);
                } else {
                    System.out.println(line);
                }
                line = br.readLine();
            }
        }

        File file = new File(outputPath);
        if (!file.exists()) {
            file.createNewFile();
        }

        FileWriter fw = new FileWriter(file.getAbsoluteFile());
        BufferedWriter bw = new BufferedWriter(fw);
        bw.write(builder.toString());
        bw.close();
    }

    public ClearNLPSentenceTagger(ClearNLPUtil parser) {
        this.parser = parser;
    }

    public static void main(String[] args) throws IOException {
        ClearNLPUtil parser = new ClearNLPUtil();

        ClearNLPSentenceTagger tagger = new ClearNLPSentenceTagger(parser);


        String trainFilePath = "/Users/Maochen/Desktop/cleanEsl.txt";
        String outputPath = "/Users/Maochen/Desktop/annotated.txt";
        tagger.processText(trainFilePath, outputPath);
    }
}
