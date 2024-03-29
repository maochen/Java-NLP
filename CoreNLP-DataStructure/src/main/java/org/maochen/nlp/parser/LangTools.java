package org.maochen.nlp.parser;

import org.apache.commons.lang3.StringUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Author: Maochen.G   contact@maochen.org License: check the LICENSE file. <p> Created by Maochen on 12/10/14.
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

    /**
     * feats string should following the pattern "k1=v1|k2=v2|k3=v3"
     */
    public static DTree getDTreeFromCoNLLXString(final String input) {
        if (input == null || input.trim().isEmpty()) {
            return null;
        }

        Map<Integer, Map<Integer, String>> semanticHeadsMap = new HashMap<>();
        String[] dNodesString = input.split(System.lineSeparator());
        DTree tree = new DTree();

        Arrays.stream(dNodesString)
                .map(s -> s.split("\t"))
                .forEachOrdered(fields -> {
                    int currentIndex = 0;
                    int id = Integer.parseInt(fields[currentIndex++]);
                    String form = fields[currentIndex++];
                    String lemma = fields[currentIndex++];
                    String cPOSTag = fields[currentIndex++];
                    String pos = fields[currentIndex++];
                    String feats = fields[currentIndex++];
                    Map<String, String> featsMap = null;
                    if (!feats.equals("_")) {
                        featsMap = Arrays.stream(feats.split("\\|"))
                                .map(entry -> entry.split("="))
                                .collect(Collectors.toMap(e -> e[0], e -> e.length > 1 ? e[1] : StringUtils.EMPTY));
                    } else {
                        featsMap = new HashMap<>();
                    }

                    String headIndex = fields[currentIndex++];
                    String depLabel = fields[currentIndex++];

                    String dump1 = fields[currentIndex++];
                    String dump2 = fields[currentIndex++];

                    String semanticHeadsString = currentIndex >= fields.length ? "_" : fields[currentIndex];

                    if (!semanticHeadsString.equals("_")) {
                        Map<Integer, String> semanticHeads = Arrays.stream(semanticHeadsString.split("\\|"))
                                .map(entry -> entry.split(":"))
                                .collect(Collectors.toMap(e -> Integer.parseInt(e[0]), e -> (e.length > 1) ? e[1] : StringUtils.EMPTY));
                        semanticHeadsMap.put(id, semanticHeads);
                    }

                    DNode node = id == 0 ? tree.getPaddingNode() : new DNode(id, form, lemma, cPOSTag, pos, depLabel);
                    if (id == 0 && !featsMap.containsKey("uuid")) {
                        featsMap.put("uuid", tree.getUUID().toString());
                    }
                    node.setFeats(featsMap);
                    node.addFeature("head", headIndex); // by the time head might not be generated!
                    if (node.getId() == 0) {
                        tree.getPaddingNode().setFeats(node.getFeats());// Substitute padding with actual 0 node from CoNLLX
                    } else {
                        tree.add(node);
                    }
                });

        tree.getPaddingNode().getFeats().remove("head");
        for (int i = 1; i < tree.size(); i++) {
            DNode node = tree.get(i);
            int headIndex = Integer.parseInt(node.getFeature("head"));
            DNode head = tree.get(headIndex);
            head.addChild(node);
            node.getFeats().remove("head");
            node.setHead(head);
        }


        // Recover semantic heads.
        semanticHeadsMap.entrySet().parallelStream().map(e -> {
            DNode node = tree.get(e.getKey());
            Map<Integer, String> nodeSemanticInfo = e.getValue();
            for (Integer id : nodeSemanticInfo.keySet()) {
                node.addSemanticHead(tree.get(id), nodeSemanticInfo.get(id));
                tree.get(id).getSemanticChildren().add(node);
            }
            return null;
        }).collect(Collectors.toList());
        return tree;
    }
}
