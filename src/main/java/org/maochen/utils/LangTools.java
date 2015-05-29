package org.maochen.utils;

import org.maochen.datastructure.DNode;
import org.maochen.datastructure.DTree;
import org.maochen.datastructure.LangLib;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by Maochen on 12/10/14.
 */
public class LangTools {

    /**
     * http://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
     */
    private static final Map<String, String> contractions = new HashMap<String, String>() {{
        put("'m", "am");
        put("'re", "are");
        put("'ve", "have");
        put("can't", "cannot");
        put("ma'am", "madam");
        put("'ll", "will");
    }};

    public static void generateLemma(DNode node) {
        // Resolve 'd
        if (node.getForm().equalsIgnoreCase("'d") && node.getPOS().equals(LangLib.POS_MD)) {
            node.setLemma(node.getLemma());
        } else if (contractions.containsKey(node.getForm())) {
            node.setLemma(contractions.get(node.getForm()));
        }
    }

    public static String getCPOSTag(String pos) {
        if (pos.equals(LangLib.POS_NNP) || pos.equals(LangLib.POS_NNPS)) {
            return LangLib.CPOSTAG_PROPN;
        } else if (pos.equals(LangLib.POS_NN) || pos.equals(LangLib.POS_NNS)) {
            return LangLib.CPOSTAG_NOUN;
        } else if (pos.startsWith(LangLib.POS_VB)) {
            return LangLib.CPOSTAG_VERB;
        } else if (pos.startsWith(LangLib.POS_JJ)) {
            return LangLib.CPOSTAG_ADJ;
        } else if (pos.equals(LangLib.POS_IN) || pos.equals(LangLib.POS_TO)) {
            return LangLib.CPOSTAG_ADP;
        } else if (pos.startsWith(LangLib.POS_RB) || pos.equals(LangLib.POS_WRB)) {
            return LangLib.CPOSTAG_ADV;
        } else if (pos.equals(LangLib.POS_MD)) {
            return LangLib.CPOSTAG_AUX;
        } else if (pos.equals(LangLib.POS_CC)) {
            return LangLib.CPOSTAG_CONJ;
        } else if (pos.equals(LangLib.POS_CD)) {
            return LangLib.CPOSTAG_NUM;
        } else if (pos.equals(LangLib.POS_DT) || pos.equals(LangLib.POS_WDT) || pos.equals(LangLib.POS_PDT) || pos.equals(LangLib.POS_EX)) {
            return LangLib.CPOSTAG_DET;
        } else if (pos.equals(LangLib.POS_POS) || pos.equals(LangLib.POS_RP)) {
            return LangLib.CPOSTAG_PART;
        } else if (pos.startsWith(LangLib.POS_PRP) || pos.startsWith(LangLib.POS_WP)) {
            return LangLib.CPOSTAG_PRON;
        } else if (pos.equals(LangLib.POS_UH)) {
            return LangLib.CPOSTAG_INTJ;
        } else if (pos.equals(LangLib.POS_WRB)) {
            return LangLib.CPOSTAG_X;
        } else if (pos.equals(LangLib.POS_SYM) || pos.equals("$") || pos.equals("#")) {
            return LangLib.CPOSTAG_SYM;
        } else if (pos.equals(".") || pos.equals(",") || pos.equals(":") || pos.equals("``") || pos.equals("''") || pos.equals(LangLib.POS_HYPH) || pos.matches("-.*B-")) {
            return LangLib.CPOSTAG_PUNCT;
        } else { // FW
            return LangLib.CPOSTAG_X;
        }
    }

    public static DTree getDTreeFromCoNLLXString(final String input) {
        if (input == null || input.trim().isEmpty()) {
            return null;
        }

        String[] dNodesString = input.split(System.lineSeparator());
        DTree tree = new DTree();

        Arrays.stream(dNodesString).parallel()
                .map(s -> s.split("\t"))
                .forEachOrdered(fields -> {
                    int currentIndex = 0;
                    int id = Integer.parseInt(fields[currentIndex++]);
                    String form = fields[currentIndex++];
                    String lemma = fields[currentIndex++];
                    String cPOSTag = fields[currentIndex++];
                    String pos = fields[currentIndex++];
                    currentIndex++;

                    String headIndex = fields[currentIndex++];
                    String depLabel = fields[currentIndex];

                    DNode node = new DNode(id, form, lemma, cPOSTag, pos, depLabel);
                    node.addFeature("head", headIndex); // by the time head might not be generated!
                    tree.add(node);
                });

        for (int i = 1; i < tree.size(); i++) {
            DNode node = tree.get(i);
            int headIndex = Integer.parseInt(node.getFeature("head"));
            DNode head = tree.get(headIndex);
            head.addChild(node);
            node.removeFeature("head");
            node.setHead(head);
        }
        return tree;
    }
}
